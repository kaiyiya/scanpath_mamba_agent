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
    指数衰减的Teacher Forcing策略

    Args:
        epoch: 当前训练轮次
        step_idx: 当前序列中的步骤索引（0-29），用于前几步保持高TF
    """
    initial_ratio = 0.7
    final_ratio = 0.2  # 从0.1提高到0.2
    decay_epochs = 150  # 从100延长到150

    # 指数衰减: ratio = 0.7 * exp(-k * epoch)
    k = -math.log(final_ratio / initial_ratio) / decay_epochs
    base_ratio = initial_ratio * math.exp(-k * epoch)
    base_ratio = max(base_ratio, final_ratio)

    # 前5步保持更高的Teacher Forcing，确保序列起始对齐
    if step_idx is not None and step_idx < 5:
        return min(base_ratio + 0.3, 0.95)

    return base_ratio


def compute_spatial_coverage_loss(pred_scanpaths):
    """合并覆盖范围、多样性和中心聚集惩罚"""
    # 覆盖范围
    pred_min = pred_scanpaths.min(dim=1)[0]
    pred_max = pred_scanpaths.max(dim=1)[0]
    pred_range = pred_max - pred_min

    coverage_x = torch.mean(((0.3 - pred_range[:, 0]).clamp(min=0.0)) ** 2)
    coverage_y = torch.mean(((0.25 - pred_range[:, 1]).clamp(min=0.0)) ** 2)

    # 多样性
    pred_mean = pred_scanpaths.mean(dim=1)
    pred_var = ((pred_scanpaths - pred_mean.unsqueeze(1)) ** 2).mean(dim=1)

    diversity_x = torch.mean(((0.015 - pred_var[:, 0]).clamp(min=0.0)) ** 2)
    diversity_y = torch.mean(((0.025 - pred_var[:, 1]).clamp(min=0.0)) ** 2)

    # Y方向中心聚集惩罚（修复：惩罚偏离0.5的任何方向）
    y_center_dist = torch.abs(pred_mean[:, 1] - 0.5)
    # 允许±0.05的偏差，超出则惩罚（修复y_mean=0.61的问题）
    y_bias_penalty = torch.mean((y_center_dist - 0.05).clamp(min=0.0) ** 2)

    # 内部加权组合
    return coverage_x + 3.0*coverage_y + diversity_x + 5.0*diversity_y + 15.0*y_bias_penalty


def compute_trajectory_smoothness_loss(pred_scanpaths, true_scanpaths):
    """合并步长、跳跃和加速度约束"""
    pred_diffs = pred_scanpaths[:, 1:] - pred_scanpaths[:, :-1]
    true_diffs = true_scanpaths[:, 1:] - true_scanpaths[:, :-1]

    pred_steps = torch.norm(pred_diffs, p=2, dim=-1)
    true_steps = torch.norm(true_diffs, p=2, dim=-1)

    # 步长匹配
    step_loss = F.mse_loss(pred_steps, true_steps)

    # 跳跃惩罚
    jump_loss = torch.mean((pred_steps - 0.2).clamp(min=0.0) ** 2)

    # 加速度约束
    if pred_steps.shape[1] > 1:
        pred_accel = pred_steps[:, 1:] - pred_steps[:, :-1]
        true_accel = true_steps[:, 1:] - true_steps[:, :-1]
        accel_loss = F.mse_loss(pred_accel, true_accel)
    else:
        accel_loss = torch.tensor(0.0, device=pred_scanpaths.device)

    return step_loss + 0.5*jump_loss + 0.3*accel_loss


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
    序列对齐损失：鼓励预测序列与真实序列在时间上对齐
    前几步给予更高权重，确保起始点和早期轨迹匹配
    """
    B, T, D = pred_scanpaths.shape

    # 计算每个预测点与对应真实点的距离
    point_distances = torch.norm(pred_scanpaths - true_scanpaths, dim=-1)  # (B, T)

    # 前几步给予更高权重（对LEV指标至关重要）
    weights = torch.ones(T, device=pred_scanpaths.device)
    weights[:5] = 5.0   # 前5步权重x5（起始点对齐）
    weights[5:10] = 3.0  # 5-10步权重x3
    weights[10:15] = 2.0  # 10-15步权重x2

    # 加权平均
    alignment_loss = torch.mean(point_distances * weights.unsqueeze(0))

    return alignment_loss


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

            # ========== 简化损失函数（13项 -> 7项）==========
            # 1. 重构损失（准确匹配真实路径）
            reconstruction_loss = nn.functional.mse_loss(predicted_scanpaths, true_scanpaths)

            # 2. KL散度正则化（防止过拟合）
            kl_loss = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
            kl_loss = kl_loss / (mus.size(0) * mus.size(1))  # 归一化

            # 3. 空间覆盖损失（合并coverage + diversity + center_penalty）
            spatial_coverage_loss = compute_spatial_coverage_loss(predicted_scanpaths)

            # 4. 轨迹平滑损失（合并step_length + jump_penalty + acceleration）
            trajectory_smoothness_loss = compute_trajectory_smoothness_loss(predicted_scanpaths, true_scanpaths)

            # 5. 方向一致性损失（合并direction + direction_continuity）
            direction_consistency_loss = compute_direction_consistency_loss(predicted_scanpaths, true_scanpaths)

            # 6. 序列对齐损失（新增：改善LEV指标）
            sequence_alignment_loss = compute_sequence_alignment_loss(predicted_scanpaths, true_scanpaths)

            # 7. 边界约束
            boundary_min = 0.02
            boundary_max = 0.98
            below_boundary = (predicted_scanpaths < boundary_min).float()
            above_boundary = (predicted_scanpaths > boundary_max).float()
            boundary_penalty = torch.mean(
                below_boundary * (boundary_min - predicted_scanpaths) ** 2 +
                above_boundary * (predicted_scanpaths - boundary_max) ** 2
            )

            # ========== 改进的权重调度 ==========
            if epoch <= 80:
                weights = {
                    'reconstruction': 2.0,  # 提高（从1.0到2.0）
                    'kl': 0.001,            # 降低（从0.005到0.001）
                    'spatial_coverage': 0.5,
                    'trajectory_smoothness': 1.5,
                    'direction_consistency': 0.5,
                    'sequence_alignment': 2.0,  # 新增：高权重改善LEV
                    'boundary': 0.2
                }
            elif epoch <= 150:
                progress = (epoch - 80) / 70.0
                weights = {
                    'reconstruction': 2.0,
                    'kl': 0.001,
                    'spatial_coverage': 0.5 + 0.3*progress,
                    'trajectory_smoothness': 1.5,
                    'direction_consistency': 0.5,
                    'sequence_alignment': 2.0 + 1.0*progress,  # 逐渐增加到3.0
                    'boundary': 0.2
                }
            else:
                weights = {
                    'reconstruction': 2.0,
                    'kl': 0.001,
                    'spatial_coverage': 0.8,
                    'trajectory_smoothness': 1.5,
                    'direction_consistency': 0.5,
                    'sequence_alignment': 3.0,  # 最终高权重
                    'boundary': 0.2
                }

            # 计算总损失
            loss = (
                weights['reconstruction'] * reconstruction_loss +
                weights['kl'] * kl_loss +
                weights['spatial_coverage'] * spatial_coverage_loss +
                weights['trajectory_smoothness'] * trajectory_smoothness_loss +
                weights['direction_consistency'] * direction_consistency_loss +
                weights['sequence_alignment'] * sequence_alignment_loss +
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
                    'SpatCov': f"{spatial_coverage_loss.item():.4f}",
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
                    # 使用较低的Teacher Forcing，更接近推理时的0.0
                    val_teacher_forcing = max(0.05, teacher_forcing_ratio * 0.3)
                    result = model(images, gt_scanpaths=true_scanpaths,
                                 teacher_forcing_ratio=val_teacher_forcing,
                                 enable_early_stop=False,
                                 use_gt_start=True)  # 验证时也使用真实起始点
                    # 安全解包：无论返回3个还是5个值，都只取前3个
                    predicted_scanpaths = result[0]
                    mus = result[1]
                    logvars = result[2]

                    # ========== 简化验证损失（与训练一致）==========
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

                    # 7. 边界约束
                    boundary_min = 0.02
                    boundary_max = 0.98
                    below_boundary = (predicted_scanpaths < boundary_min).float()
                    above_boundary = (predicted_scanpaths > boundary_max).float()
                    boundary_penalty = torch.mean(
                        below_boundary * (boundary_min - predicted_scanpaths) ** 2 +
                        above_boundary * (predicted_scanpaths - boundary_max) ** 2
                    )

                    # 使用与训练相同的权重
                    if epoch <= 80:
                        weights = {
                            'reconstruction': 2.0,
                            'kl': 0.001,
                            'spatial_coverage': 0.5,
                            'trajectory_smoothness': 1.5,
                            'direction_consistency': 0.5,
                            'sequence_alignment': 2.0,
                            'boundary': 0.2
                        }
                    elif epoch <= 150:
                        progress = (epoch - 80) / 70.0
                        weights = {
                            'reconstruction': 2.0,
                            'kl': 0.001,
                            'spatial_coverage': 0.5 + 0.3*progress,
                            'trajectory_smoothness': 1.5,
                            'direction_consistency': 0.5,
                            'sequence_alignment': 2.0 + 1.0*progress,
                            'boundary': 0.2
                        }
                    else:
                        weights = {
                            'reconstruction': 2.0,
                            'kl': 0.001,
                            'spatial_coverage': 0.8,
                            'trajectory_smoothness': 1.5,
                            'direction_consistency': 0.5,
                            'sequence_alignment': 3.0,
                            'boundary': 0.2
                        }

                    # 计算总损失
                    loss = (
                        weights['reconstruction'] * reconstruction_loss +
                        weights['kl'] * kl_loss +
                        weights['spatial_coverage'] * spatial_coverage_loss +
                        weights['trajectory_smoothness'] * trajectory_smoothness_loss +
                        weights['direction_consistency'] * direction_consistency_loss +
                        weights['sequence_alignment'] * sequence_alignment_loss +
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

        # 学习率衰减（每个epoch结束后）
        scheduler.step()

    print("\n训练完成！")
    print(f"最佳验证损失: {best_loss:.4f}")
    print(f"\n下一步：使用 visualize_mamba_agent.py 可视化结果")


if __name__ == '__main__':
    train()
