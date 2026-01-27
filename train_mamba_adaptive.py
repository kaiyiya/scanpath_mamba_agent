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
import math


def compute_teacher_forcing_ratio(epoch, step_idx=None):
    """
    优化的Teacher Forcing策略（改善训练稳定性）
    
    改进点：
    1. 更慢的衰减速度，避免训练不稳定
    2. 保持较高的最终比例，减少训练和推理差异
    3. 更平滑的步级衰减

    Args:
        epoch: 当前训练轮次
        step_idx: 当前序列中的步骤索引（0-29），用于前几步保持高TF
    """
    initial_ratio = 0.9  # 提高初始比例，确保早期训练稳定
    final_ratio = 0.4  # 提高最终比例（从0.2到0.4），减少训练和推理差异
    decay_epochs = 50  # 50 epoch

    # 线性衰减（更稳定）：ratio = initial - (initial - final) * (epoch / decay_epochs)
    base_ratio = initial_ratio - (initial_ratio - final_ratio) * min(epoch / decay_epochs, 1.0)
    base_ratio = max(base_ratio, final_ratio)

    # 前几步平滑衰减（更保守的策略）
    if step_idx is not None:
        if step_idx < 3:
            # 前3步：额外+0.1，确保起始稳定
            return min(base_ratio + 0.1, 0.95)
        elif step_idx < 6:
            # 3-6步：额外+0.05
            return min(base_ratio + 0.05, 0.90)
        elif step_idx < 10:
            # 6-10步：额外+0.02
            return min(base_ratio + 0.02, 0.85)

    return base_ratio


def compute_spatial_coverage_loss(pred_scanpaths):
    """合并覆盖范围、多样性和中心聚集惩罚（改进版：提高Y方向覆盖）"""
    # 覆盖范围
    pred_min = pred_scanpaths.min(dim=1)[0]
    pred_max = pred_scanpaths.max(dim=1)[0]
    pred_range = pred_max - pred_min

    # 提高覆盖目标：X方向0.5，Y方向0.5（之前是0.3和0.25）
    coverage_x = torch.mean(((0.5 - pred_range[:, 0]).clamp(min=0.0)) ** 2)
    coverage_y = torch.mean(((0.5 - pred_range[:, 1]).clamp(min=0.0)) ** 2)

    # 多样性
    pred_mean = pred_scanpaths.mean(dim=1)
    pred_var = ((pred_scanpaths - pred_mean.unsqueeze(1)) ** 2).mean(dim=1)

    diversity_x = torch.mean(((0.015 - pred_var[:, 0]).clamp(min=0.0)) ** 2)
    diversity_y = torch.mean(((0.025 - pred_var[:, 1]).clamp(min=0.0)) ** 2)

    # Y方向中心聚集惩罚（修复：惩罚偏离0.5的任何方向）
    y_center_dist = torch.abs(pred_mean[:, 1] - 0.5)
    # 允许±0.1的偏差（放宽限制），超出则惩罚
    y_bias_penalty = torch.mean((y_center_dist - 0.1).clamp(min=0.0) ** 2)

    # 内部加权组合（降低y_bias_penalty权重，给Y方向更多自由）
    return coverage_x + 3.0 * coverage_y + diversity_x + 5.0 * diversity_y + 5.0 * y_bias_penalty


def compute_trajectory_smoothness_loss(pred_scanpaths, true_scanpaths):
    """合并步长、跳跃和加速度约束（修复：移除过度的跳跃惩罚）"""
    pred_diffs = pred_scanpaths[:, 1:] - pred_scanpaths[:, :-1]
    true_diffs = true_scanpaths[:, 1:] - true_scanpaths[:, :-1]

    pred_steps = torch.norm(pred_diffs, p=2, dim=-1)
    true_steps = torch.norm(true_diffs, p=2, dim=-1)

    # 步长匹配
    step_loss = F.mse_loss(pred_steps, true_steps)

    # 跳跃惩罚（修复：提高阈值到0.5，允许更大的步长）
    # 之前0.2太小，导致路径移动距离过短
    jump_loss = torch.mean((pred_steps - 0.5).clamp(min=0.0) ** 2)

    # 加速度约束
    if pred_steps.shape[1] > 1:
        pred_accel = pred_steps[:, 1:] - pred_steps[:, :-1]
        true_accel = true_steps[:, 1:] - true_steps[:, :-1]
        accel_loss = F.mse_loss(pred_accel, true_accel)
    else:
        accel_loss = torch.tensor(0.0, device=pred_scanpaths.device)

    # 降低jump_loss权重（从0.5到0.1）
    return step_loss + 0.1 * jump_loss + 0.3 * accel_loss


def compute_direction_consistency_loss(pred_scanpaths, true_scanpaths):
    """合并方向变化和方向连续性"""
    pred_diffs = pred_scanpaths[:, 1:] - pred_scanpaths[:, :-1]
    true_diffs = true_scanpaths[:, 1:] - true_scanpaths[:, :-1]

    pred_steps = torch.norm(pred_diffs, p=2, dim=-1, keepdim=True)
    true_steps = torch.norm(true_diffs, p=2, dim=-1, keepdim=True)

    pred_directions = pred_diffs / (pred_steps + 1e-8)
    true_directions = true_diffs / (true_steps + 1e-8)

    # 方向变化
    if pred_directions.shape[1] > 1:
        pred_dir_diffs = pred_directions[:, 1:] - pred_directions[:, :-1]
        true_dir_diffs = true_directions[:, 1:] - true_directions[:, :-1]
        direction_loss = F.mse_loss(
            torch.norm(pred_dir_diffs, p=2, dim=-1),
            torch.norm(true_dir_diffs, p=2, dim=-1)
        )
    else:
        direction_loss = torch.tensor(0.0, device=pred_scanpaths.device)

    # 方向连续性
    if pred_directions.shape[1] > 0:
        pred_similarity = F.cosine_similarity(pred_directions[:, :-1], pred_directions[:, 1:], dim=-1)
        true_similarity = F.cosine_similarity(true_directions[:, :-1], true_directions[:, 1:], dim=-1)
        continuity_loss = F.mse_loss(pred_similarity, true_similarity)
    else:
        continuity_loss = torch.tensor(0.0, device=pred_scanpaths.device)

    return direction_loss + continuity_loss


def compute_sequence_alignment_loss(pred_scanpaths, true_scanpaths):
    """
    完整序列对齐损失：约束所有30步（方案A - 精确复制，修复版）

    关键改进：
    - 约束所有30步，确保完整序列对齐
    - 降低内部权重，避免过度关注前几步导致路径"卡住"
    - 目标：让模型学会"精确复制"真实路径的完整轨迹
    """
    B, T, D = pred_scanpaths.shape

    # 计算所有时间步的点对点距离
    point_distances = torch.norm(pred_scanpaths - true_scanpaths, dim=-1)  # (B, T)

    # 权重配置：降低权重，避免过度约束（之前权重太高导致模型"卡住"）
    weights = torch.ones(T, device=pred_scanpaths.device)
    weights[:5] = 3.0  # 前5步：权重3.0（从15.0降低）
    weights[5:10] = 2.5  # 5-10步：权重2.5（从10.0降低）
    weights[10:15] = 2.0  # 10-15步：权重2.0（从8.0降低）
    weights[15:20] = 1.5  # 15-20步：权重1.5（从6.0降低）
    weights[20:25] = 1.3  # 20-25步：权重1.3（从5.0降低）
    weights[25:] = 1.2  # 25-30步：权重1.2（从4.0降低）

    # 计算所有30步的加权平均
    alignment_loss = torch.mean(point_distances * weights.unsqueeze(0))

    return alignment_loss


def compute_motion_consistency_loss(pred_scanpaths, true_scanpaths):
    """
    运动一致性损失（方案C-改进版）：选择性约束方向和步长

    改进：只约束"合理"的运动，避免过度约束导致N形路径

    包含两个部分：
    1. 方向相似度损失：使用余弦相似度约束运动方向
    2. 步长相似度损失：使用MSE约束运动步长

    关键改进：
    - 只对步长在合理范围内的运动进行约束
    - 对于过小的步长（< 0.01），不约束方向（避免噪声）
    - 对于过大的步长（> 0.3），降低约束权重（允许探索）

    Args:
        pred_scanpaths: 预测路径 (B, T, 2)
        true_scanpaths: 真实路径 (B, T, 2)

    Returns:
        motion_loss: 运动一致性损失标量
    """
    # 计算运动向量（相邻点之间的位移）
    pred_motions = pred_scanpaths[:, 1:] - pred_scanpaths[:, :-1]  # (B, T-1, 2)
    true_motions = true_scanpaths[:, 1:] - true_scanpaths[:, :-1]  # (B, T-1, 2)

    # 计算步长
    pred_step_lengths = torch.norm(pred_motions, p=2, dim=-1)  # (B, T-1)
    true_step_lengths = torch.norm(true_motions, p=2, dim=-1)  # (B, T-1)

    # 1. 方向相似度损失（余弦相似度）- 选择性约束
    # 归一化运动向量得到方向
    pred_directions = F.normalize(pred_motions, p=2, dim=-1, eps=1e-8)  # (B, T-1, 2)
    true_directions = F.normalize(true_motions, p=2, dim=-1, eps=1e-8)  # (B, T-1, 2)

    # 计算余弦相似度
    cosine_similarity = (pred_directions * true_directions).sum(dim=-1)  # (B, T-1)

    # 选择性约束：只对合理步长的运动约束方向
    # 步长太小（< 0.01）：可能是噪声，不约束
    # 步长太大（> 0.3）：可能是探索性跳跃，降低约束
    step_mask = (true_step_lengths > 0.01) & (true_step_lengths < 0.3)  # (B, T-1)

    # 对于大步长，使用较小的权重
    large_step_mask = true_step_lengths >= 0.3
    large_step_weight = 0.3  # 大步长的方向约束权重降低到30%

    # 计算加权方向损失
    direction_loss_per_step = 1.0 - cosine_similarity  # (B, T-1)
    direction_loss_weighted = torch.where(
        step_mask,
        direction_loss_per_step,  # 正常步长：全权重
        torch.where(
            large_step_mask,
            direction_loss_per_step * large_step_weight,  # 大步长：降低权重
            torch.zeros_like(direction_loss_per_step)  # 小步长：不约束
        )
    )
    direction_loss = direction_loss_weighted.mean()

    # 2. 步长相似度损失（MSE）- 使用相对误差而不是绝对误差
    # 改进：使用相对误差，避免对大步长过度惩罚
    # 相对误差 = |pred - true| / (true + eps)
    step_relative_error = torch.abs(pred_step_lengths - true_step_lengths) / (true_step_lengths + 1e-6)

    # 只对合理步长约束
    step_loss_per_step = step_relative_error ** 2
    step_loss_weighted = torch.where(
        step_mask,
        step_loss_per_step,
        torch.zeros_like(step_loss_per_step)
    )
    step_length_loss = step_loss_weighted.mean()

    # 3. 组合损失（方向和步长同等重要）
    motion_loss = direction_loss + step_length_loss

    return motion_loss


# 已移除 compute_batch_diversity_loss 函数
# 方案A：专注于精确复制路径，不鼓励多样性


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

    # 学习率调度器 - 使用带warmup的余弦退火（改善训练稳定性）
    # 优化：添加warmup阶段，避免早期训练不稳定
    warmup_epochs = 5
    
    # 使用SequentialLR组合warmup和余弦退火
    # Warmup调度器（用于前warmup_epochs个epoch）
    def lr_lambda_warmup(epoch):
        return (epoch + 1) / warmup_epochs  # 线性warmup
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_warmup)
    
    # 余弦退火调度器（用于warmup之后的epoch）
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs - warmup_epochs,
        eta_min=config.learning_rate * 0.05  # 最小学习率为初始的5%（提高，避免学习率过小）
    )
    
    # 组合调度器
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # 早停机制：基于验证位置误差而不是损失
    # 优化：增加patience，避免过早停止
    best_val_position_error = float('inf')
    patience_counter = 0
    early_stopping_patience = 15  # 增加到15，给模型更多训练机会
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
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'=' * 80}")

        # 训练
        model.train()
        epoch_loss = 0
        epoch_position_error = 0

        train_bar = tqdm(train_loader, desc="训练")
        for batch_idx, batch in enumerate(train_bar):
            images = batch['image'].to(config.device)
            true_scanpaths = batch['scanpath'].to(config.device)

            # 前向传播 - 传递真实位置用于Teacher Forcing
            # 改进Teacher Forcing策略：指数衰减
            teacher_forcing_ratio = compute_teacher_forcing_ratio(epoch)

            # 训练时显式设置enable_early_stop=False，确保返回3个值
            # use_gt_start=True 确保使用真实起始点，改善LEV指标
            predicted_scanpaths, mus, logvars = model(
                images,
                gt_scanpaths=true_scanpaths,
                teacher_forcing_ratio=teacher_forcing_ratio,
                enable_early_stop=False,
                use_gt_start=True  # 使用真实起始点
            )

            # ========== 方案A：精确复制路径的损失函数 ==========
            # 1. 重构损失（准确匹配真实路径）- 提高权重
            reconstruction_loss = nn.functional.mse_loss(predicted_scanpaths, true_scanpaths)

            # 2. KL散度正则化（降低权重，减少随机性）
            kl_loss = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
            kl_loss = kl_loss / (mus.size(0) * mus.size(1))  # 归一化

            # 3. 空间覆盖损失（保持适度约束）
            spatial_coverage_loss = compute_spatial_coverage_loss(predicted_scanpaths)

            # 4. 轨迹平滑损失（保持轨迹平滑）
            trajectory_smoothness_loss = compute_trajectory_smoothness_loss(predicted_scanpaths, true_scanpaths)

            # 5. 方向一致性损失（保持方向一致）
            direction_consistency_loss = compute_direction_consistency_loss(predicted_scanpaths, true_scanpaths)

            # 6. 序列对齐损失（约束所有30步，大幅提高权重）
            sequence_alignment_loss = compute_sequence_alignment_loss(predicted_scanpaths, true_scanpaths)

            # 7. 运动一致性损失（方案C：显式约束方向和步长）
            motion_consistency_loss = compute_motion_consistency_loss(predicted_scanpaths, true_scanpaths)

            # 8. 边界约束
            boundary_min = 0.02
            boundary_max = 0.98
            below_boundary = (predicted_scanpaths < boundary_min).float()
            above_boundary = (predicted_scanpaths > boundary_max).float()
            boundary_penalty = torch.mean(
                below_boundary * (boundary_min - predicted_scanpaths) ** 2 +
                above_boundary * (predicted_scanpaths - boundary_max) ** 2
            )

            # ========== 优化版损失权重：平衡各项损失，改善训练稳定性 ==========
            # 改进策略：
            # 1. 降低sequence_alignment权重，避免过度约束导致路径"卡住"
            # 2. 提高motion_consistency权重，改善序列连续性
            # 3. 平衡reconstruction和sequence_alignment，避免冲突
            # 4. 适度提高KL权重，增加模型多样性
            # 5. 渐进式权重调整，避免训练不稳定
            if epoch <= 15:
                # 早期：重点学习基本位置预测
                weights = {
                    'reconstruction': 5.0,  # 提高基础重建损失
                    'kl': 0.001,  # 适度增加KL，保持多样性
                    'spatial_coverage': 1.5,  # 降低，避免过度约束
                    'trajectory_smoothness': 0.3,  # 提高，改善平滑性
                    'direction_consistency': 0.3,  # 提高，改善方向一致性
                    'sequence_alignment': 3.0,  # 降低，避免过度约束
                    'motion_consistency': 0.5,  # 大幅提高，改善运动连续性
                    'boundary': 0.3
                }
            elif epoch <= 30:
                # 中期：平衡各项损失
                progress = (epoch - 15) / 15.0
                weights = {
                    'reconstruction': 5.0 - 1.0 * progress,  # 逐渐降低到4.0
                    'kl': 0.001 + 0.001 * progress,  # 逐渐增加到0.002
                    'spatial_coverage': 1.5 + 0.5 * progress,  # 逐渐增加到2.0
                    'trajectory_smoothness': 0.3 + 0.2 * progress,  # 逐渐增加到0.5
                    'direction_consistency': 0.3 + 0.2 * progress,  # 逐渐增加到0.5
                    'sequence_alignment': 3.0 + 1.0 * progress,  # 逐渐增加到4.0
                    'motion_consistency': 0.5 + 0.3 * progress,  # 逐渐增加到0.8
                    'boundary': 0.3
                }
            else:
                # 后期：精细调优
                weights = {
                    'reconstruction': 4.0,  # 最终权重
                    'kl': 0.002,  # 最终权重（适度增加多样性）
                    'spatial_coverage': 2.0,  # 最终权重
                    'trajectory_smoothness': 0.5,  # 最终权重
                    'direction_consistency': 0.5,  # 最终权重
                    'sequence_alignment': 4.0,  # 最终权重（降低，避免过度约束）
                    'motion_consistency': 0.8,  # 最终权重（提高，改善连续性）
                    'boundary': 0.3
                }

            # 计算总损失（添加motion_consistency项）
            loss = (
                    weights['reconstruction'] * reconstruction_loss +
                    weights['kl'] * kl_loss +
                    weights['spatial_coverage'] * spatial_coverage_loss +
                    weights['trajectory_smoothness'] * trajectory_smoothness_loss +
                    weights['direction_consistency'] * direction_consistency_loss +
                    weights['sequence_alignment'] * sequence_alignment_loss +
                    weights['motion_consistency'] * motion_consistency_loss +
                    weights['boundary'] * boundary_penalty
            )

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
                train_bar.set_postfix({
                    'Loss': f"{avg_loss:.4f}",
                    'PosErr': f"{avg_error:.4f}",
                    'TF': f"{teacher_forcing_ratio:.3f}",
                    'SeqAlign': f"{sequence_alignment_loss.item():.4f}",
                    'Recon': f"{reconstruction_loss.item():.4f}",
                    'KL': f"{kl_loss.item():.5f}",
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
                    # 优化：验证时使用与训练更接近的Teacher Forcing，减少分布差异
                    val_teacher_forcing = max(0.3, teacher_forcing_ratio * 0.7)  # 提高验证时TF比例
                    result = model(images, gt_scanpaths=true_scanpaths,
                                   teacher_forcing_ratio=val_teacher_forcing,
                                   enable_early_stop=False,
                                   use_gt_start=True)  # 验证时也使用真实起始点
                    # 安全解包：无论返回3个还是5个值，都只取前3个
                    predicted_scanpaths = result[0]
                    mus = result[1]
                    logvars = result[2]

                    # ========== 方案A验证损失（与训练一致）==========
                    # 1. 重构损失
                    reconstruction_loss = nn.functional.mse_loss(predicted_scanpaths, true_scanpaths)

                    # 2. KL散度正则化
                    kl_loss = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
                    kl_loss = kl_loss / (mus.size(0) * mus.size(1))

                    # 3. 空间覆盖损失
                    spatial_coverage_loss = compute_spatial_coverage_loss(predicted_scanpaths)

                    # 4. 轨迹平滑损失
                    trajectory_smoothness_loss = compute_trajectory_smoothness_loss(predicted_scanpaths, true_scanpaths)

                    # 5. 方向一致性损失
                    direction_consistency_loss = compute_direction_consistency_loss(predicted_scanpaths, true_scanpaths)

                    # 6. 序列对齐损失
                    sequence_alignment_loss = compute_sequence_alignment_loss(predicted_scanpaths, true_scanpaths)

                    # 7. 运动一致性损失
                    motion_consistency_loss = compute_motion_consistency_loss(predicted_scanpaths, true_scanpaths)

                    # 8. 边界约束
                    boundary_min = 0.02
                    boundary_max = 0.98
                    below_boundary = (predicted_scanpaths < boundary_min).float()
                    above_boundary = (predicted_scanpaths > boundary_max).float()
                    boundary_penalty = torch.mean(
                        below_boundary * (boundary_min - predicted_scanpaths) ** 2 +
                        above_boundary * (predicted_scanpaths - boundary_max) ** 2
                    )

                    # 使用与训练相同的权重（优化版）
                    if epoch <= 15:
                        weights = {
                            'reconstruction': 5.0,
                            'kl': 0.001,
                            'spatial_coverage': 1.5,
                            'trajectory_smoothness': 0.3,
                            'direction_consistency': 0.3,
                            'sequence_alignment': 3.0,
                            'motion_consistency': 0.5,
                            'boundary': 0.3
                        }
                    elif epoch <= 30:
                        progress = (epoch - 15) / 15.0
                        weights = {
                            'reconstruction': 5.0 - 1.0 * progress,
                            'kl': 0.001 + 0.001 * progress,
                            'spatial_coverage': 1.5 + 0.5 * progress,
                            'trajectory_smoothness': 0.3 + 0.2 * progress,
                            'direction_consistency': 0.3 + 0.2 * progress,
                            'sequence_alignment': 3.0 + 1.0 * progress,
                            'motion_consistency': 0.5 + 0.3 * progress,
                            'boundary': 0.3
                        }
                    else:
                        weights = {
                            'reconstruction': 4.0,
                            'kl': 0.002,
                            'spatial_coverage': 2.0,
                            'trajectory_smoothness': 0.5,
                            'direction_consistency': 0.5,
                            'sequence_alignment': 4.0,
                            'motion_consistency': 0.8,
                            'boundary': 0.3
                        }

                    # 计算总损失（包含motion_consistency项）
                    loss = (
                            weights['reconstruction'] * reconstruction_loss +
                            weights['kl'] * kl_loss +
                            weights['spatial_coverage'] * spatial_coverage_loss +
                            weights['trajectory_smoothness'] * trajectory_smoothness_loss +
                            weights['direction_consistency'] * direction_consistency_loss +
                            weights['sequence_alignment'] * sequence_alignment_loss +
                            weights['motion_consistency'] * motion_consistency_loss +
                            weights['boundary'] * boundary_penalty
                    )

                    # 计算位置误差
                    position_error = torch.norm(
                        predicted_scanpaths - true_scanpaths,
                        dim=-1
                    ).mean()

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

        # 学习率调度（每个epoch结束后）
        scheduler.step()

    print("\n训练完成！")
    print(f"最佳验证损失: {best_loss:.4f}")
    print(f"\n下一步：使用 visualize_mamba_agent.py 可视化结果")


if __name__ == '__main__':
    train()
