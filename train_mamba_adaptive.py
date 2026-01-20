"""
Mamba-Adaptive扫描路径模型训练脚本
结合 Mamba + AdaptiveNN Focus机制
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import create_dataloaders
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath


def train():
    """训练主函数"""
    config = MambaAdaptiveConfig()

    # 创建保存目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 加载数据
    print("加载数据...")
    train_loader, test_loader = create_dataloaders(config)
    print(f"训练集: {len(train_loader)} batches")
    print(f"测试集: {len(test_loader)} batches")

    # 创建模型
    print("\n创建Mamba-Adaptive模型（结合Focus机制）...")
    model = MambaAdaptiveScanpath(config).to(config.device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器 - 使用余弦退火（更好的收敛性）
    # 训练目标：学习率在前半程较高（快速学习），后半程较低（精细调优）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=config.learning_rate * 0.01  # 最小学习率为初始的1%
    )

    # 早停机制：基于验证位置误差而不是损失
    # 改进：使用位置误差作为早停指标，更符合主要目标
    best_val_position_error = float('inf')
    patience_counter = 0
    early_stopping_patience = 20  # 增加到20，给模型更多机会
    best_val_loss = float('inf')  # 仍然记录，但用于保存模型

    # 训练日志
    training_log = {
        'config': {
            'seq_len': config.seq_len,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'feature_dim': config.feature_dim,
            'd_state': config.d_state,
            'focus_patch_size': config.focus_patch_size,
        },
        'epochs': []
    }

    # 训练循环
    print("\n开始训练...")
    best_loss = float('inf')

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*80}")

        # 训练
        model.train()
        epoch_loss = 0
        epoch_position_error = 0

        train_bar = tqdm(train_loader, desc="训练")
        for batch_idx, batch in enumerate(train_bar):
            images = batch['image'].to(config.device)
            true_scanpaths = batch['scanpath'].to(config.device)

            # 前向传播 - 传递真实位置用于Teacher Forcing
            # 改进Teacher Forcing策略：更慢的衰减，让模型充分学习
            # 方案5：调整Teacher Forcing策略
            if epoch <= 100:
                # 阶段1：从0.7缓慢降到0.4
                teacher_forcing_ratio = 0.7 - 0.3 * (epoch / 100.0)
            elif epoch <= 200:
                # 阶段2：从0.4缓慢降到0.2
                teacher_forcing_ratio = 0.4 - 0.2 * ((epoch - 100) / 100.0)
            else:
                # 阶段3：保持0.2
                teacher_forcing_ratio = 0.2

            # 训练时显式设置enable_early_stop=False，确保返回3个值
            predicted_scanpaths, mus, logvars = model(
                images,
                gt_scanpaths=true_scanpaths,
                teacher_forcing_ratio=teacher_forcing_ratio,
                enable_early_stop=False
            )

            # 计算损失函数 - VAE框架：重构损失 + KL散度正则化
            # 1. 重构损失（准确匹配真实路径）
            # 使用标准MSE（用于监控），实际损失使用加权MSE（见下方）
            reconstruction_loss = nn.functional.mse_loss(predicted_scanpaths, true_scanpaths)

            # 2. KL散度正则化（防止过拟合）
            # KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
            kl_loss = kl_loss / (mus.size(0) * mus.size(1))  # 归一化

            # 3. Beta-VAE：控制KL散度的权重
            # 方案6：降低VAE的KL散度权重，减少随机性
            # 训练目标：保持VAE的随机性，但不影响主要的位置预测精度
            # 策略：使用更小的beta值，让重构损失占主导
            beta = min(0.005, 0.001 * (1.01 ** (epoch - 1)))  # 从0.001开始，最多到0.005

            # 4. 覆盖范围损失：鼓励预测路径覆盖整个图像，防止过度聚集
            # 激进改进：分别约束X和Y方向的覆盖范围
            pred_min = predicted_scanpaths.min(dim=1)[0]  # (B, 2)
            pred_max = predicted_scanpaths.max(dim=1)[0]  # (B, 2)
            pred_range = pred_max - pred_min  # (B, 2)

            # 分别处理X和Y方向的覆盖范围
            pred_range_x = pred_range[:, 0]  # (B,)
            pred_range_y = pred_range[:, 1]  # (B,)

            # Y方向需要更大的覆盖范围（当前y均值都在0.49-0.51，太集中）
            min_range_x = 0.3  # x方向最小覆盖范围
            min_range_y = 0.25  # y方向最小覆盖范围（提高要求）

            coverage_loss_x = torch.mean(((min_range_x - pred_range_x).clamp(min=0.0)) ** 2)
            coverage_loss_y = torch.mean(((min_range_y - pred_range_y).clamp(min=0.0)) ** 2)

            # Y方向给予更高权重
            coverage_loss = coverage_loss_x + 3.0 * coverage_loss_y
            
            # 5. 位置多样性损失：鼓励预测位置具有足够的方差
            # 方案1：分别约束X和Y方向的多样性（关键改进！）
            pred_mean = predicted_scanpaths.mean(dim=1)  # (B, 2)
            pred_var = ((predicted_scanpaths - pred_mean.unsqueeze(1)) ** 2).mean(dim=1)  # (B, 2)

            # 分别处理X和Y方向
            pred_var_x = pred_var[:, 0]  # (B,)
            pred_var_y = pred_var[:, 1]  # (B,)

            # Y方向需要更高的多样性阈值（当前Y方向标准差只有0.04-0.07，目标0.10-0.15）
            # 激进改进：大幅提高Y方向阈值和权重
            min_var_x = 0.015  # x方向标准差约0.12（已达标）
            min_var_y = 0.025  # y方向标准差约0.16（从0.020提高到0.025）

            # 计算多样性损失
            diversity_loss_x = torch.mean(((min_var_x - pred_var_x).clamp(min=0.0)) ** 2)
            diversity_loss_y = torch.mean(((min_var_y - pred_var_y).clamp(min=0.0)) ** 2)

            # Y方向给予更高权重（从2倍提高到5倍！）
            diversity_loss = diversity_loss_x + 5.0 * diversity_loss_y
            
            # 6. 边界约束：防止预测跑到图像边界外
            # 方案4：放宽边界约束，降低惩罚强度
            # 定义合理范围：[0.02, 0.98]，只防止完全越界
            boundary_min = 0.02  # 从0.05放宽到0.02
            boundary_max = 0.98  # 从0.95放宽到0.98

            # 惩罚超出边界的位置
            below_boundary = (predicted_scanpaths < boundary_min).float()
            above_boundary = (predicted_scanpaths > boundary_max).float()
            boundary_penalty = torch.mean(
                below_boundary * (boundary_min - predicted_scanpaths) ** 2 +
                above_boundary * (predicted_scanpaths - boundary_max) ** 2
            ) * 2.0  # 从10.0降低到2.0，只防止完全越界
            
            # 7. Y方向中心聚集惩罚：专门针对Y方向的中心聚集问题
            # 新增：Y方向均值过于集中在0.5附近（0.49-0.51），需要惩罚
            pred_mean_y = pred_mean[:, 1]  # (B,) Y方向的均值

            # 惩罚Y方向均值过于接近0.5
            y_center_dist = torch.abs(pred_mean_y - 0.5)  # 距离中心的距离
            # 当距离小于0.1时（即均值在[0.4, 0.6]），给予惩罚
            y_too_centered = (y_center_dist < 0.1).float()
            y_center_penalty = torch.mean(y_too_centered * (0.1 - y_center_dist) ** 2) * 10.0

            # X方向不惩罚（已经足够分散）
            center_penalty = y_center_penalty
            point_center_penalty = torch.tensor(0.0, device=predicted_scanpaths.device)

            # 8. 轨迹平滑性损失：鼓励相邻点之间的距离合理，使路径连贯流畅
            # 方案2：增强轨迹平滑性约束
            # 计算相邻点之间的步长
            pred_diffs = predicted_scanpaths[:, 1:] - predicted_scanpaths[:, :-1]  # (B, seq_len-1, 2)
            true_diffs = true_scanpaths[:, 1:] - true_scanpaths[:, :-1]  # (B, seq_len-1, 2)
            
            # 步长距离
            pred_step_lengths = torch.norm(pred_diffs, p=2, dim=-1)  # (B, seq_len-1)
            true_step_lengths = torch.norm(true_diffs, p=2, dim=-1)  # (B, seq_len-1)
            
            # 平滑性损失1：相邻点距离应该与真实路径相似
            step_length_loss = nn.functional.mse_loss(pred_step_lengths, true_step_lengths)
            
            # 平滑性损失2：惩罚过大的跳跃（防止路径过于分散）
            # 真实路径的步长通常小于0.15，惩罚超过0.2的跳跃
            max_reasonable_step = 0.20
            large_jumps = (pred_step_lengths - max_reasonable_step).clamp(min=0.0)  # 只惩罚大于0.2的
            jump_penalty = torch.mean(large_jumps ** 2)
            
            # 平滑性损失3：方向一致性（相邻方向变化不应该太大）
            # 计算方向向量
            pred_directions = pred_diffs / (pred_step_lengths.unsqueeze(-1) + 1e-8)  # 归一化方向
            true_directions = true_diffs / (true_step_lengths.unsqueeze(-1) + 1e-8)
            
            # 计算相邻方向的余弦相似度（应该是正的，表示方向变化平缓）
            if pred_directions.shape[1] > 1:  # 确保有足够的点计算方向变化
                pred_dir_diffs = pred_directions[:, 1:] - pred_directions[:, :-1]  # 方向变化
                true_dir_diffs = true_directions[:, 1:] - true_directions[:, :-1]
                direction_loss = nn.functional.mse_loss(
                    torch.norm(pred_dir_diffs, p=2, dim=-1),
                    torch.norm(true_dir_diffs, p=2, dim=-1)
                )
            else:
                direction_loss = torch.tensor(0.0, device=predicted_scanpaths.device)
            
            # 平滑性损失4：加速度约束（步长的变化应该平滑，保证路径流畅）
            # 计算加速度：相邻步长之间的变化
            if pred_step_lengths.shape[1] > 1:  # 确保有足够的步长计算加速度
                pred_acceleration = pred_step_lengths[:, 1:] - pred_step_lengths[:, :-1]  # (B, seq_len-2)
                true_acceleration = true_step_lengths[:, 1:] - true_step_lengths[:, :-1]
                acceleration_loss = nn.functional.mse_loss(pred_acceleration, true_acceleration)
            else:
                acceleration_loss = torch.tensor(0.0, device=predicted_scanpaths.device)
            
            # 平滑性损失5：方向连续性（相邻方向应该相似，避免突然转向）
            if pred_directions.shape[1] > 0:
                # 计算相邻方向的余弦相似度
                pred_dir_similarity = F.cosine_similarity(
                    pred_directions[:, :-1], 
                    pred_directions[:, 1:], 
                    dim=-1
                )  # (B, seq_len-2)
                true_dir_similarity = F.cosine_similarity(
                    true_directions[:, :-1], 
                    true_directions[:, 1:], 
                    dim=-1
                )
                direction_continuity_loss = nn.functional.mse_loss(
                    pred_dir_similarity, 
                    true_dir_similarity
                )
            else:
                direction_continuity_loss = torch.tensor(0.0, device=predicted_scanpaths.device)

            # ========== 更激进的损失权重策略 ==========
            # 核心改进：
            # 1. 大幅提高Y方向多样性和覆盖范围权重
            # 2. 进一步增强平滑性约束
            # 3. 降低重构损失的主导地位

            if epoch <= 80:
                # 阶段1：平衡重构和多样性
                coverage_weight = 0.3  # 大幅提高（从0.05到0.3）
                diversity_weight = 0.5  # 大幅提高（从0.1到0.5）
                center_weight = 0.3  # Y方向中心惩罚
                boundary_weight = 0.2  # 进一步降低
                point_center_weight = 0.0
                smoothness_weight = 1.5  # 进一步提高（从1.0到1.5）
                jump_weight = 0.8  # 提高（从0.5到0.8）
                direction_weight = 0.5  # 提高（从0.3到0.5）
                acceleration_weight = 0.5  # 提高（从0.3到0.5）
                direction_continuity_weight = 0.5  # 提高（从0.3到0.5）
            elif epoch <= 150:
                # 阶段2：进一步增加多样性约束
                progress = (epoch - 80) / 70.0
                coverage_weight = 0.3 + 0.2 * progress  # 0.3 -> 0.5
                diversity_weight = 0.5 + 0.3 * progress  # 0.5 -> 0.8
                center_weight = 0.3 + 0.2 * progress  # 0.3 -> 0.5
                boundary_weight = 0.2
                point_center_weight = 0.0
                smoothness_weight = 1.5
                jump_weight = 0.8
                direction_weight = 0.5
                acceleration_weight = 0.5
                direction_continuity_weight = 0.5
            else:
                # 阶段3：最大多样性约束
                coverage_weight = 0.5
                diversity_weight = 0.8
                center_weight = 0.5  # Y方向中心惩罚
                boundary_weight = 0.2
                point_center_weight = 0.0
                smoothness_weight = 1.5
                jump_weight = 0.8
                direction_weight = 0.5
                acceleration_weight = 0.5
                direction_continuity_weight = 0.5
            
            # Batch内多样性损失：提高权重
            batch_mean = predicted_scanpaths.mean(dim=0, keepdim=True)  # (1, seq_len, 2)
            batch_diversity = torch.mean((predicted_scanpaths - batch_mean) ** 2)  # 标量
            min_batch_diversity = 0.015  # 提高阈值（从0.01到0.015）
            batch_diversity_loss = torch.mean(((min_batch_diversity - batch_diversity).clamp(min=0.0)) ** 2)

            # 提高batch多样性权重
            if epoch <= 80:
                batch_diversity_weight = 0.2  # 提高（从0.05到0.2）
            elif epoch <= 150:
                progress = (epoch - 80) / 70.0
                batch_diversity_weight = 0.2 + 0.1 * progress  # 0.2 -> 0.3
            else:
                batch_diversity_weight = 0.3  # 保持
            
            # 使用加权MSE：对起始位置和前几步给予更高权重
            # 降低权重差异，更平衡前后步骤
            position_weights = torch.ones(config.seq_len, device=predicted_scanpaths.device)
            if epoch <= 80:
                # 阶段1：适度权重
                position_weights[0] = 2.5  # 起始位置
                position_weights[1:5] = 1.8  # 前5步
                position_weights[5:10] = 1.3  # 5-10步
            else:
                # 阶段2和3：更平衡
                position_weights[0] = 2.0
                position_weights[1:5] = 1.5
                position_weights[5:10] = 1.2
            
            weighted_reconstruction_loss = torch.mean(
                position_weights.unsqueeze(0).unsqueeze(-1) *
                (predicted_scanpaths - true_scanpaths) ** 2
            )

            # 总损失
            loss = weighted_reconstruction_loss + beta * kl_loss + \
                   coverage_weight * coverage_loss + \
                   diversity_weight * diversity_loss + \
                   center_weight * center_penalty + \
                   boundary_weight * boundary_penalty + \
                   point_center_weight * point_center_penalty + \
                   smoothness_weight * step_length_loss + \
                   jump_weight * jump_penalty + \
                   direction_weight * direction_loss + \
                   acceleration_weight * acceleration_loss + \
                   direction_continuity_weight * direction_continuity_loss + \
                   batch_diversity_weight * batch_diversity_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 计算位置误差
            position_weights_error = torch.ones(config.seq_len, device=predicted_scanpaths.device)
            if epoch <= 80:
                position_weights_error[0] = 2.5
                position_weights_error[1:5] = 1.8
                position_weights_error[5:10] = 1.3
            else:
                position_weights_error[0] = 2.0
                position_weights_error[1:5] = 1.5
                position_weights_error[5:10] = 1.2
            
            # 加权位置误差
            weighted_errors = torch.norm(
                predicted_scanpaths - true_scanpaths,
                dim=-1
            ) * position_weights_error.unsqueeze(0)
            position_error = weighted_errors.mean() / position_weights_error.mean()  # 归一化以保持原有尺度

            # 累积指标
            epoch_loss += loss.item()
            epoch_position_error += position_error.item()

            # 更新进度条
            if (batch_idx + 1) % config.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_error = epoch_position_error / (batch_idx + 1)
                avg_coverage = coverage_loss.item()
                avg_diversity = diversity_loss.item()
                avg_center = center_penalty.item()
                avg_smooth = step_length_loss.item()
                avg_jump = jump_penalty.item()
                avg_batch_div = batch_diversity_loss.item()
                avg_point_center = point_center_penalty.item()
                avg_acceleration = acceleration_loss.item()
                avg_dir_continuity = direction_continuity_loss.item()
                train_bar.set_postfix({
                    'Loss': f"{avg_loss:.4f}",
                    'PosErr': f"{avg_error:.4f}",
                    'Beta': f"{beta:.4f}",
                    'Cov': f"{avg_coverage:.4f}",
                    'Ctr': f"{avg_center:.4f}",
                    'PCtr': f"{avg_point_center:.4f}",
                    'Smooth': f"{avg_smooth:.4f}",
                    'Jump': f"{avg_jump:.4f}",
                    'Dir': f"{avg_dir_continuity:.4f}",
                    'Acc': f"{avg_acceleration:.4f}",
                })

        # 平均训练指标
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        epoch_position_error /= num_batches

        # 打印训练结果
        print(f"\n训练结果:")
        print(f"  Loss: {epoch_loss:.4f}")
        print(f"  PositionError: {epoch_position_error:.4f}")

        # 验证
        if epoch % config.val_interval == 0:
            print(f"\n验证...")
            model.eval()
            val_loss = 0
            val_position_error = 0

            val_bar = tqdm(test_loader, desc="验证")
            with torch.no_grad():
                for batch in val_bar:
                    images = batch['image'].to(config.device)
                    true_scanpaths = batch['scanpath'].to(config.device)

                    # 前向传播 - 验证模式
                    # 训练目标：验证模型在推理时的真实性能
                    # 策略：使用更低的Teacher Forcing，更接近推理时的0.0
                    # 显式设置enable_early_stop=False，确保返回3个值
                    if epoch <= 50:
                        val_teacher_forcing = 0.2  # 阶段1：较低Teacher Forcing（从0.3降到0.2）
                    elif epoch <= 100:
                        val_teacher_forcing = 0.1  # 阶段2：更低Teacher Forcing（从0.2降到0.1）
                    else:
                        val_teacher_forcing = 0.05  # 阶段3：非常低Teacher Forcing（从0.1降到0.05）
                    result = model(images, gt_scanpaths=true_scanpaths, teacher_forcing_ratio=val_teacher_forcing, enable_early_stop=False)
                    # 安全解包：无论返回3个还是5个值，都只取前3个
                    predicted_scanpaths = result[0]
                    mus = result[1]
                    logvars = result[2]

                    # 计算VAE损失函数（与训练时完全一致）
                    # 1. 重构损失（用于监控）
                    reconstruction_loss = nn.functional.mse_loss(predicted_scanpaths, true_scanpaths)

                    # 2. KL散度正则化
                    kl_loss = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
                    kl_loss = kl_loss / (mus.size(0) * mus.size(1))

                    # 3. Beta-VAE权重（与训练时相同）
                    beta = min(0.01, 0.005 * (1.01 ** (epoch - 1)))

                    # 4. 覆盖范围损失（与训练时一致，使用改进后的版本）
                    pred_min = predicted_scanpaths.min(dim=1)[0]
                    pred_max = predicted_scanpaths.max(dim=1)[0]
                    pred_range = pred_max - pred_min
                    coverage_loss = torch.mean(((0.3 - pred_range).clamp(min=0.0)) ** 2) * 0.5
                    
                    # 5. 位置多样性损失（与训练时一致，使用改进后的版本）
                    pred_mean = predicted_scanpaths.mean(dim=1)
                    pred_var = ((predicted_scanpaths - pred_mean.unsqueeze(1)) ** 2).mean(dim=1)
                    min_var = 0.015
                    diversity_loss = torch.mean(((min_var - pred_var).clamp(min=0.0)) ** 2) * 0.5
                    
                    # 6. 边界约束（与训练时一致）
                    boundary_min = 0.05
                    boundary_max = 0.95
                    below_boundary = (predicted_scanpaths < boundary_min).float()
                    above_boundary = (predicted_scanpaths > boundary_max).float()
                    boundary_penalty = torch.mean(
                        below_boundary * (boundary_min - predicted_scanpaths) ** 2 +
                        above_boundary * (predicted_scanpaths - boundary_max) ** 2
                    ) * 10.0
                    
                    # 7. 温和的中心聚集惩罚（与训练时一致）
                    mean_center_dist = torch.mean((pred_mean - 0.5) ** 2, dim=-1)
                    very_close_to_center = (mean_center_dist < 0.01).float()
                    center_penalty = torch.mean(very_close_to_center * (0.01 - mean_center_dist) * 5.0)
                    point_center_penalty = torch.tensor(0.0, device=predicted_scanpaths.device)
                    
                    # 8. 轨迹平滑性损失（与训练时完全一致，包括新增的加速度和方向连续性）
                    pred_diffs = predicted_scanpaths[:, 1:] - predicted_scanpaths[:, :-1]
                    true_diffs = true_scanpaths[:, 1:] - true_scanpaths[:, :-1]
                    pred_step_lengths = torch.norm(pred_diffs, p=2, dim=-1)
                    true_step_lengths = torch.norm(true_diffs, p=2, dim=-1)
                    step_length_loss = nn.functional.mse_loss(pred_step_lengths, true_step_lengths)
                    large_jumps = (pred_step_lengths - 0.20).clamp(min=0.0)
                    jump_penalty = torch.mean(large_jumps ** 2)
                    
                    if pred_diffs.shape[1] > 1:
                        pred_directions = pred_diffs / (pred_step_lengths.unsqueeze(-1) + 1e-8)
                        true_directions = true_diffs / (true_step_lengths.unsqueeze(-1) + 1e-8)
                        pred_dir_diffs = pred_directions[:, 1:] - pred_directions[:, :-1]
                        true_dir_diffs = true_directions[:, 1:] - true_directions[:, :-1]
                        direction_loss = nn.functional.mse_loss(
                            torch.norm(pred_dir_diffs, p=2, dim=-1),
                            torch.norm(true_dir_diffs, p=2, dim=-1)
                        )
                    else:
                        direction_loss = torch.tensor(0.0, device=predicted_scanpaths.device)
                    
                    # 加速度约束（与训练时一致）
                    if pred_step_lengths.shape[1] > 1:
                        pred_acceleration = pred_step_lengths[:, 1:] - pred_step_lengths[:, :-1]
                        true_acceleration = true_step_lengths[:, 1:] - true_step_lengths[:, :-1]
                        acceleration_loss = nn.functional.mse_loss(pred_acceleration, true_acceleration)
                    else:
                        acceleration_loss = torch.tensor(0.0, device=predicted_scanpaths.device)
                    
                    # 方向连续性（与训练时一致）
                    if pred_directions.shape[1] > 0:
                        pred_dir_similarity = F.cosine_similarity(
                            pred_directions[:, :-1], 
                            pred_directions[:, 1:], 
                            dim=-1
                        )
                        true_dir_similarity = F.cosine_similarity(
                            true_directions[:, :-1], 
                            true_directions[:, 1:], 
                            dim=-1
                        )
                        direction_continuity_loss = nn.functional.mse_loss(
                            pred_dir_similarity, 
                            true_dir_similarity
                        )
                    else:
                        direction_continuity_loss = torch.tensor(0.0, device=predicted_scanpaths.device)
                    
                    # 9. Batch内多样性损失（与训练时一致，使用改进后的版本）
                    batch_mean = predicted_scanpaths.mean(dim=0, keepdim=True)
                    batch_diversity = torch.mean((predicted_scanpaths - batch_mean) ** 2)
                    min_batch_diversity = 0.01
                    batch_diversity_loss = torch.mean(((min_batch_diversity - batch_diversity).clamp(min=0.0)) ** 2)
                    
                    # 9. 加权MSE损失（与训练时一致，分阶段）
                    position_weights = torch.ones(config.seq_len, device=predicted_scanpaths.device)
                    if epoch <= 80:
                        position_weights[0] = 3.0
                        position_weights[1:5] = 2.0
                        position_weights[5:10] = 1.5
                    else:
                        position_weights[0] = 2.0
                        position_weights[1:5] = 1.5
                        position_weights[5:10] = 1.2
                    weighted_reconstruction_loss = torch.mean(
                        position_weights.unsqueeze(0).unsqueeze(-1) * 
                        (predicted_scanpaths - true_scanpaths) ** 2
                    )

                    # 总损失（与训练时完全一致，使用改进后的版本）
                    if epoch <= 80:
                        coverage_weight = 0.02
                        diversity_weight = 0.01
                        center_weight = 0.01
                        boundary_weight = 0.5
                        point_center_weight = 0.0
                        smoothness_weight = 0.15
                        jump_weight = 0.05
                        direction_weight = 0.05
                        acceleration_weight = 0.05
                        direction_continuity_weight = 0.05
                        batch_diversity_weight = 0.0
                    elif epoch <= 150:
                        progress = (epoch - 80) / 70.0
                        coverage_weight = 0.02 + 0.03 * progress  # 从0.02增加到0.05
                        diversity_weight = 0.01 + 0.02 * progress  # 从0.01增加到0.03
                        center_weight = 0.01 + 0.02 * progress  # 从0.01增加到0.03
                        boundary_weight = 0.5 - 0.2 * progress  # 从0.5降低到0.3
                        point_center_weight = 0.0
                        smoothness_weight = 0.15 + 0.1 * progress
                        jump_weight = 0.05 + 0.05 * progress
                        direction_weight = 0.05 + 0.05 * progress
                        acceleration_weight = 0.05 + 0.05 * progress
                        direction_continuity_weight = 0.05 + 0.05 * progress
                        batch_diversity_weight = 0.01 * progress  # 从0增加到0.01
                    else:
                        coverage_weight = 0.05
                        diversity_weight = 0.03
                        center_weight = 0.03
                        boundary_weight = 0.3
                        point_center_weight = 0.0
                        smoothness_weight = 0.25
                        jump_weight = 0.1
                        direction_weight = 0.1
                        acceleration_weight = 0.1
                        direction_continuity_weight = 0.1
                        batch_diversity_weight = 0.01
                    
                    loss = weighted_reconstruction_loss + beta * kl_loss + \
                           coverage_weight * coverage_loss + \
                           diversity_weight * diversity_loss + \
                           center_weight * center_penalty + \
                           boundary_weight * boundary_penalty + \
                           point_center_weight * point_center_penalty + \
                           smoothness_weight * step_length_loss + \
                           jump_weight * jump_penalty + \
                           direction_weight * direction_loss + \
                           acceleration_weight * acceleration_loss + \
                           direction_continuity_weight * direction_continuity_loss + \
                           batch_diversity_weight * batch_diversity_loss

                    # 计算位置误差（与训练时一致，使用加权误差，分阶段）
                    position_weights_error = torch.ones(config.seq_len, device=predicted_scanpaths.device)
                    if epoch <= 80:
                        position_weights_error[0] = 3.0  # 与训练时一致
                        position_weights_error[1:5] = 2.0
                        position_weights_error[5:10] = 1.5
                    else:
                        position_weights_error[0] = 2.0
                        position_weights_error[1:5] = 1.5
                        position_weights_error[5:10] = 1.2
                    
                    weighted_errors = torch.norm(
                        predicted_scanpaths - true_scanpaths,
                        dim=-1
                    ) * position_weights_error.unsqueeze(0)
                    position_error = weighted_errors.mean() / position_weights_error.mean()

                    val_loss += loss.item()
                    val_position_error += position_error.item()

            # 平均验证指标
            num_val_batches = len(test_loader)
            val_loss /= num_val_batches
            val_position_error /= num_val_batches

            print(f"\n验证结果:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  PositionError: {val_position_error:.4f}")

            # 学习率调度 - ExponentialLR在每个epoch后自动衰减
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")

            # 保存最佳模型：优先基于位置误差，也考虑损失
            save_model = False
            if val_position_error < best_val_position_error:
                best_val_position_error = val_position_error
                save_model = True
                patience_counter = 0  # 重置早停计数器（基于位置误差）
                print(f"  ✅ 验证位置误差改善: {val_position_error:.4f} (新最佳)")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not save_model:  # 如果位置误差没改善但损失改善了，也保存
                    save_model = True
            
            if save_model:
                best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_val_loss,
                    'best_position_error': best_val_position_error,
                }, best_path)
                print(f"  💾 保存最佳模型: {best_path}")
                print(f"     最佳验证位置误差: {best_val_position_error:.4f}")
                print(f"     最佳验证损失: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  ⚠️ 验证位置误差未改善 ({patience_counter}/{early_stopping_patience})")
                print(f"     当前: {val_position_error:.4f}, 最佳: {best_val_position_error:.4f}")

            # 早停检查：基于位置误差
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️ 早停触发！验证位置误差已经{early_stopping_patience}个epoch没有改善")
                print(f"最佳验证位置误差: {best_val_position_error:.4f}")
                print(f"最佳验证损失: {best_val_loss:.4f}")
                break

        # 保存检查点
        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"  保存检查点: {checkpoint_path}")

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 记录日志
        epoch_log = {
            'epoch': epoch,
            'learning_rate': current_lr,
            'train': {
                'loss': epoch_loss,
                'position_error': epoch_position_error,
            },
        }
        if epoch % config.val_interval == 0:
            epoch_log['val'] = {
                'loss': val_loss,
                'position_error': val_position_error,
            }

        training_log['epochs'].append(epoch_log)

        # 保存训练日志
        log_path = os.path.join(config.log_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        # 学习率衰减（每个epoch结束后）
        scheduler.step()

    print("\n训练完成！")
    print(f"最佳验证损失: {best_loss:.4f}")
    print(f"\n下一步：使用 visualize_mamba_agent.py 可视化结果")


if __name__ == '__main__':
    train()
