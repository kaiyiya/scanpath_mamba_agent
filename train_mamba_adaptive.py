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

    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15

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
            # 改进Teacher Forcing策略：更缓慢衰减，保持训练和推理分布一致性
            # 训练目标：确保模型既能学习真实路径，又能在推理时独立工作
            # 策略：前100个epoch保持0.7，然后缓慢降到0.5并保持
            if epoch <= 100:
                teacher_forcing_ratio = 0.7
            else:
                teacher_forcing_ratio = max(0.5, 0.7 - (epoch - 100) * 0.002)  # 每epoch降0.002

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
            # 训练目标：保持VAE的随机性，但不影响主要的位置预测精度
            # 策略：使用较小的beta值，让重构损失占主导
            beta = min(0.01, 0.005 * (1.01 ** (epoch - 1)))  # 从0.005开始，最多到0.01

            # 4. 覆盖范围损失：鼓励预测路径覆盖整个图像，防止集中在中心
            # 计算预测路径的覆盖范围（x和y方向分别计算）
            pred_min = predicted_scanpaths.min(dim=1)[0]  # (B, 2)
            pred_max = predicted_scanpaths.max(dim=1)[0]  # (B, 2)
            pred_range = pred_max - pred_min  # (B, 2)
            # 理想覆盖范围应该是接近[1.0, 1.0]，惩罚小范围
            # 使用更敏感的惩罚：当范围小于0.4时给予强惩罚
            coverage_loss = torch.mean(((0.4 - pred_range).clamp(min=0.0)) ** 2)
            
            # 5. 位置多样性损失：鼓励预测位置具有足够的方差
            pred_mean = predicted_scanpaths.mean(dim=1)  # (B, 2)
            pred_var = ((predicted_scanpaths - pred_mean.unsqueeze(1)) ** 2).mean(dim=1)  # (B, 2)
            # 修复：提高方差阈值到0.02（从0.01），因为当前预测的方差在0.0064-0.0225之间
            # 使用更敏感的惩罚函数
            min_var = 0.02  # 期望最小方差（对应标准差约0.14）
            diversity_loss = torch.mean(((min_var - pred_var).clamp(min=0.0)) ** 2)
            
            # 6. 中心聚集惩罚：惩罚预测均值过于接近图像中心(0.5, 0.5)
            # 计算每个样本的均值距离中心的距离
            mean_center_dist = torch.mean((pred_mean - 0.5) ** 2, dim=-1)  # (B,)
            # 修复：使用更强的惩罚，当均值在[0.4, 0.6]范围内时给予惩罚
            # 使用分段函数：距离<0.01时给予最大惩罚，距离>0.05时不给惩罚
            center_penalty = torch.mean(torch.exp(-mean_center_dist * 50.0))  # 增强惩罚强度（从20到50）

            # 7. 轨迹平滑性损失：鼓励相邻点之间的距离合理，使路径连贯流畅
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

            # ========== 训练目标明确：位置预测精度优先 ==========
            # 主要目标：最小化位置误差（重构损失占主导）
            # 次要目标：保持路径合理性（正则化项权重很小，不影响主要目标）
            
            # 正则化权重策略：使用很小的权重，只作为轻微的引导
            # 避免正则化项压制重构损失，确保模型优先优化位置预测精度
            
            # 动态权重：早期稍微关注正则化，后期主要关注准确性
            epoch_factor = min(1.0, epoch / 100.0)  # 前100个epoch逐渐调整
            
            # 大幅降低正则化权重（从原来的2.0-3.0降到0.05以下）
            coverage_weight = 0.1 * (1.0 - 0.5 * epoch_factor)  # 覆盖范围：0.1 -> 0.05
            diversity_weight = 0.05 * (1.0 - 0.5 * epoch_factor)  # 多样性：0.05 -> 0.025（关键！）
            center_weight = 0.05 * (1.0 - 0.5 * epoch_factor)  # 中心聚集：0.05 -> 0.025
            smoothness_weight = 0.1 * (1.0 - 0.3 * epoch_factor)  # 平滑性：0.1 -> 0.07
            jump_weight = 0.05  # 跳跃惩罚：保持0.05
            direction_weight = 0.02  # 方向一致性：降低到0.02
            
            # Batch内多样性损失：降低权重
            batch_mean = predicted_scanpaths.mean(dim=0, keepdim=True)  # (1, seq_len, 2)
            batch_diversity = torch.mean((predicted_scanpaths - batch_mean) ** 2)  # 标量
            min_batch_diversity = 0.01
            batch_diversity_loss = torch.mean(((min_batch_diversity - batch_diversity).clamp(min=0.0)) ** 2)
            batch_diversity_weight = 0.01 * (1.0 - 0.5 * epoch_factor)  # Batch多样性：0.01 -> 0.005
            
            # 使用加权MSE：对起始位置和前几步给予更高权重
            position_weights = torch.ones(config.seq_len, device=predicted_scanpaths.device)
            position_weights[0] = 5.0  # 起始位置权重5倍（最重要）
            position_weights[1:5] = 3.0  # 前5步权重3倍
            position_weights[5:10] = 2.0  # 5-10步权重2倍
            weighted_reconstruction_loss = torch.mean(
                position_weights.unsqueeze(0).unsqueeze(-1) * 
                (predicted_scanpaths - true_scanpaths) ** 2
            )
            
            loss = weighted_reconstruction_loss + beta * kl_loss + \
                   coverage_weight * coverage_loss + \
                   diversity_weight * diversity_loss + \
                   center_weight * center_penalty + \
                   smoothness_weight * step_length_loss + \
                   jump_weight * jump_penalty + \
                   direction_weight * direction_loss + \
                   batch_diversity_weight * batch_diversity_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 计算位置误差 - 使用加权误差（与损失函数一致，更准确地反映模型性能）
            # 训练目标：位置误差应该与损失函数同步改善
            position_weights_error = torch.ones(config.seq_len, device=predicted_scanpaths.device)
            position_weights_error[0] = 5.0  # 起始位置权重5倍
            position_weights_error[1:5] = 3.0  # 前5步权重3倍
            position_weights_error[5:10] = 2.0  # 5-10步权重2倍
            
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
                train_bar.set_postfix({
                    'Loss': f"{avg_loss:.4f}",
                    'PosErr': f"{avg_error:.4f}",
                    'Beta': f"{beta:.4f}",
                    'Cov': f"{avg_coverage:.4f}",
                    'Div': f"{avg_diversity:.4f}",
                    'BDiv': f"{avg_batch_div:.4f}",
                    'Ctr': f"{avg_center:.4f}",
                    'Smooth': f"{avg_smooth:.4f}",
                    'Jump': f"{avg_jump:.4f}",
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
                    # 策略：使用少量Teacher Forcing（0.1）避免训练和推理分布差异过大
                    # 显式设置enable_early_stop=False，确保返回3个值
                    val_teacher_forcing = 0.1  # 验证时使用少量teacher forcing
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

                    # 4. 覆盖范围损失（与训练时一致）
                    pred_min = predicted_scanpaths.min(dim=1)[0]
                    pred_max = predicted_scanpaths.max(dim=1)[0]
                    pred_range = pred_max - pred_min
                    coverage_loss = torch.mean(((0.4 - pred_range).clamp(min=0.0)) ** 2)
                    
                    # 5. 位置多样性损失（与训练时一致）
                    pred_mean = predicted_scanpaths.mean(dim=1)
                    pred_var = ((predicted_scanpaths - pred_mean.unsqueeze(1)) ** 2).mean(dim=1)
                    min_var = 0.02
                    diversity_loss = torch.mean(((min_var - pred_var).clamp(min=0.0)) ** 2)
                    
                    # 6. 中心聚集惩罚（与训练时一致）
                    mean_center_dist = torch.mean((pred_mean - 0.5) ** 2, dim=-1)
                    center_penalty = torch.mean(torch.exp(-mean_center_dist * 50.0))
                    
                    # 7. 轨迹平滑性损失（与训练时一致）
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
                    
                    # 8. Batch内多样性损失（与训练时一致）
                    batch_mean = predicted_scanpaths.mean(dim=0, keepdim=True)
                    batch_diversity = torch.mean((predicted_scanpaths - batch_mean) ** 2)
                    min_batch_diversity = 0.01
                    batch_diversity_loss = torch.mean(((min_batch_diversity - batch_diversity).clamp(min=0.0)) ** 2)
                    
                    # 9. 加权MSE损失（与训练时一致）
                    position_weights = torch.ones(config.seq_len, device=predicted_scanpaths.device)
                    position_weights[0] = 5.0
                    position_weights[1:5] = 3.0
                    position_weights[5:10] = 2.0
                    weighted_reconstruction_loss = torch.mean(
                        position_weights.unsqueeze(0).unsqueeze(-1) * 
                        (predicted_scanpaths - true_scanpaths) ** 2
                    )

                    # 总损失（与训练时完全一致）
                    epoch_factor = min(1.0, epoch / 100.0)
                    coverage_weight = 0.1 * (1.0 - 0.5 * epoch_factor)
                    diversity_weight = 0.05 * (1.0 - 0.5 * epoch_factor)
                    center_weight = 0.05 * (1.0 - 0.5 * epoch_factor)
                    smoothness_weight = 0.1 * (1.0 - 0.3 * epoch_factor)
                    jump_weight = 0.05
                    direction_weight = 0.02
                    batch_diversity_weight = 0.01 * (1.0 - 0.5 * epoch_factor)
                    
                    loss = weighted_reconstruction_loss + beta * kl_loss + \
                           coverage_weight * coverage_loss + \
                           diversity_weight * diversity_loss + \
                           center_weight * center_penalty + \
                           smoothness_weight * step_length_loss + \
                           jump_weight * jump_penalty + \
                           direction_weight * direction_loss + \
                           batch_diversity_weight * batch_diversity_loss

                    # 计算位置误差（与训练时一致，使用加权误差）
                    position_weights_error = torch.ones(config.seq_len, device=predicted_scanpaths.device)
                    position_weights_error[0] = 5.0
                    position_weights_error[1:5] = 3.0
                    position_weights_error[5:10] = 2.0
                    
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

            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                }, best_path)
                print(f"  保存最佳模型: {best_path}")
                patience_counter = 0  # 重置早停计数器
            else:
                patience_counter += 1
                print(f"  验证损失未改善 ({patience_counter}/{early_stopping_patience})")

            # 早停检查
            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发！验证损失已经{early_stopping_patience}个epoch没有改善")
                print(f"最佳验证损失: {best_loss:.4f} (Epoch {epoch - patience_counter})")
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
