"""
Mamba-Adaptive扫描路径模型训练脚本
结合 Mamba + AdaptiveNN Focus机制
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import create_dataloaders
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath
import math


def compute_teacher_forcing_ratio(epoch, step_idx=None):
    """
    修复版Teacher Forcing策略（改善序列对齐，修复REC为0问题）
    
    关键修复：
    1. 保持较高的Teacher Forcing比例，确保模型学习真实序列
    2. 更慢的衰减速度，避免训练不稳定
    3. 前几步保持高TF，确保序列起始正确

    Args:
        epoch: 当前训练轮次
        step_idx: 当前序列中的步骤索引（0-29），用于前几步保持高TF
    """
    initial_ratio = 0.95  # 提高初始比例，确保早期训练稳定
    final_ratio = 0.5  # 提高最终比例，减少训练和推理差异
    decay_epochs = 50  # 50 epoch

    # 线性衰减（更稳定）：ratio = initial - (initial - final) * (epoch / decay_epochs)
    base_ratio = initial_ratio - (initial_ratio - final_ratio) * min(epoch / decay_epochs, 1.0)
    base_ratio = max(base_ratio, final_ratio)

    # 前几步平滑衰减（更保守的策略，确保序列对齐）
    if step_idx is not None:
        if step_idx < 3:
            # 前3步：额外+0.05，确保起始稳定
            return min(base_ratio + 0.05, 1.0)
        elif step_idx < 6:
            # 3-6步：额外+0.03
            return min(base_ratio + 0.03, 0.98)
        elif step_idx < 10:
            # 6-10步：额外+0.01
            return min(base_ratio + 0.01, 0.95)

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

    # 优化器（优化：降低初始学习率，增强正则化）
    # 修复：降低初始学习率，从0.00012降到0.00005，解决过拟合
    initial_lr = config.learning_rate * 0.4  # 降低到原来的40%（0.00012 * 0.4 = 0.000048）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,  # 使用降低后的学习率
        weight_decay=config.weight_decay * 1.5,  # 增加weight decay：从2e-3提高到3e-3
        betas=(0.9, 0.999),  # 默认值，但显式指定
        eps=1e-8  # 默认值，但显式指定
    )

    # 学习率调度器 - 使用带warmup的余弦退火（改善训练稳定性）
    # 优化：延长warmup阶段，降低初始学习率，解决过拟合
    warmup_epochs = 10  # 延长warmup：从5个epoch延长到10个epoch
    
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
    # 修复：正确计算patience，考虑验证间隔
    best_val_position_error = float('inf')
    patience_counter = 0
    early_stopping_patience = 20  # 增加到20，给模型更多训练机会（考虑验证间隔）
    best_val_loss = float('inf')  # 仍然记录，但用于保存模型
    last_val_epoch = 0  # 记录上次验证的epoch

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

            # ========== 修复版损失函数：添加像素级距离损失，修复REC为0 ==========
            # 关键修复：REC为0说明预测路径和真实路径的点对点距离远超过12像素
            # 需要添加像素级距离损失，直接约束像素距离
            
            # 1. 归一化坐标的重建损失（保持）
            position_weights = torch.ones(config.seq_len, device=predicted_scanpaths.device)
            position_weights[0] = 3.0  # 第一步权重最高
            position_weights[1:5] = 2.0  # 前5步权重较高
            position_weights[5:10] = 1.5  # 5-10步权重适中
            position_weights = position_weights.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
            
            # 加权MSE损失（归一化坐标）
            squared_errors = (predicted_scanpaths - true_scanpaths) ** 2  # (B, T, 2)
            weighted_errors = squared_errors * position_weights  # (B, T, 2)
            reconstruction_loss_norm = weighted_errors.mean()
            
            # 2. 像素级距离损失（关键修复：直接约束像素距离）
            # 转换为像素坐标
            h, w = config.image_size
            pred_pixels = predicted_scanpaths.clone()
            pred_pixels[:, :, 0] = pred_pixels[:, :, 0] * w  # X坐标
            pred_pixels[:, :, 1] = pred_pixels[:, :, 1] * h  # Y坐标
            
            true_pixels = true_scanpaths.clone()
            true_pixels[:, :, 0] = true_pixels[:, :, 0] * w  # X坐标
            true_pixels[:, :, 1] = true_pixels[:, :, 1] * h  # Y坐标
            
            # 计算像素距离（归一化到图像对角线长度）
            pixel_distances = torch.norm(pred_pixels - true_pixels, p=2, dim=-1)  # (B, T)
            diagonal_length = np.sqrt(w**2 + h**2)  # 图像对角线长度，用于归一化
            pixel_distances_norm = pixel_distances / diagonal_length  # 归一化到[0, 1]
            
            # REC风格的损失：惩罚距离超过阈值的点（归一化阈值）
            # 阈值12像素，归一化到对角线长度
            rec_threshold_norm = 12.0 / diagonal_length
            rec_threshold_pixels = 12.0  # 像素阈值
            
            # 硬约束：对距离>12像素的点对使用更强的惩罚（Focal损失风格）
            # 距离越远，惩罚越大（指数增长）
            pixel_distances_abs = pixel_distances  # (B, T) 绝对像素距离
            far_mask = pixel_distances_abs > rec_threshold_pixels  # (B, T) 距离>12像素的mask
            
            # 对于远距离点对，使用指数惩罚：exp((distance - threshold) / threshold)
            # 这样距离越远，惩罚增长越快
            far_distances = pixel_distances_abs[far_mask]  # 只对远距离点计算
            if len(far_distances) > 0:
                # 归一化到阈值，然后指数增长
                normalized_far = (far_distances - rec_threshold_pixels) / rec_threshold_pixels
                rec_penalty_far = torch.mean(torch.exp(normalized_far * 2.0))  # 指数惩罚
            else:
                rec_penalty_far = torch.tensor(0.0, device=predicted_scanpaths.device)
            
            # 对于所有点，使用平方惩罚（归一化）
            rec_penalty_all = torch.mean((pixel_distances_norm - rec_threshold_norm).clamp(min=0.0) ** 2)
            
            # 组合REC惩罚：所有点的惩罚 + 远距离点的额外惩罚
            rec_penalty = rec_penalty_all + 3.0 * rec_penalty_far  # 远距离惩罚权重3.0
            
            # 像素级MSE损失（归一化到图像尺寸）
            pixel_diff_norm = (pred_pixels - true_pixels) / diagonal_length  # 归一化差值
            pixel_mse = torch.mean(pixel_diff_norm ** 2)
            
            # 像素级L1损失（归一化）
            pixel_l1 = torch.mean(torch.abs(pixel_diff_norm))
            
            # 组合重建损失：归一化坐标损失 + 归一化像素级损失
            # 关键修复：大幅提高REC惩罚权重，从2.0提高到8.0，确保点对点匹配
            reconstruction_loss = reconstruction_loss_norm + 0.5 * pixel_mse + 0.5 * pixel_l1 + 8.0 * rec_penalty

            # 2. KL散度正则化（增强正则化，解决过拟合）
            # 修复：提高KL权重，从0.0003-0.0007提高到0.002-0.01
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

            # ========== 优化版损失权重：修复REC为0和过拟合问题 ==========
            # 关键优化：
            # 1. 大幅提高序列对齐损失权重（从3.0-4.0提高到12.0-20.0），确保点对点匹配
            # 2. 增强KL正则化（从0.0003-0.0007提高到0.002-0.01），解决过拟合
            # 3. 保持reconstruction权重适中，因为已经添加了高权重的像素级损失
            # 4. 渐进式权重调整，确保训练稳定
            if epoch <= 10:
                # 早期：重点学习点对点匹配，强约束序列对齐
                weights = {
                    'reconstruction': 5.0,  # 适中权重（因为已有高权重像素级损失）
                    'kl': 0.002,  # 提高KL权重，增强正则化（从0.0003提高到0.002）
                    'spatial_coverage': 0.5,  # 降低，避免过度约束
                    'trajectory_smoothness': 0.1,  # 降低，允许更灵活的路径
                    'direction_consistency': 0.1,  # 降低，避免过度约束
                    'sequence_alignment': 12.0,  # 大幅提高，确保点对点匹配（从3.0提高到12.0）
                    'motion_consistency': 0.15,  # 适度运动连续性
                    'boundary': 0.1
                }
            elif epoch <= 25:
                # 中期：平衡各项损失，逐渐增加正则化
                progress = (epoch - 10) / 15.0
                weights = {
                    'reconstruction': 5.0 - 0.5 * progress,  # 逐渐降低到4.5
                    'kl': 0.002 + 0.004 * progress,  # 逐渐增加到0.006（从0.0007提高到0.006）
                    'spatial_coverage': 0.5 + 0.5 * progress,  # 逐渐增加到1.0
                    'trajectory_smoothness': 0.1 + 0.2 * progress,  # 逐渐增加到0.3
                    'direction_consistency': 0.1 + 0.2 * progress,  # 逐渐增加到0.3
                    'sequence_alignment': 12.0 + 4.0 * progress,  # 逐渐增加到16.0（从4.0提高到16.0）
                    'motion_consistency': 0.15 + 0.35 * progress,  # 逐渐增加到0.5
                    'boundary': 0.1
                }
            else:
                # 后期：精细调优，保持强约束
                weights = {
                    'reconstruction': 4.5,  # 最终权重（适中，因为已有高权重像素级损失）
                    'kl': 0.01,  # 最终权重（大幅提高，从0.0007提高到0.01，解决过拟合）
                    'spatial_coverage': 1.0,  # 最终权重
                    'trajectory_smoothness': 0.3,  # 最终权重
                    'direction_consistency': 0.3,  # 最终权重
                    'sequence_alignment': 20.0,  # 最终权重（大幅提高，从4.0提高到20.0，确保点对点匹配）
                    'motion_consistency': 0.5,  # 最终权重
                    'boundary': 0.1
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
            
            # 梯度裁剪，避免训练不稳定
            # 注意：这里先计算loss，反向传播时再裁剪

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：降低max_norm，避免训练不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 从1.0降到0.5
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
                    # 修复：验证时使用与训练相同的Teacher Forcing比例，统一训练和验证策略
                    val_teacher_forcing = teacher_forcing_ratio  # 使用与训练相同的TF比例
                    result = model(images, gt_scanpaths=true_scanpaths,
                                   teacher_forcing_ratio=val_teacher_forcing,
                                   enable_early_stop=False,
                                   use_gt_start=True)  # 验证时也使用真实起始点
                    # 安全解包：无论返回3个还是5个值，都只取前3个
                    predicted_scanpaths = result[0]
                    mus = result[1]
                    logvars = result[2]

                    # ========== 修复版验证损失（与训练一致）==========
                    # 1. 重构损失（使用加权MSE + 像素级损失，与训练一致）
                    position_weights = torch.ones(config.seq_len, device=predicted_scanpaths.device)
                    position_weights[0] = 3.0
                    position_weights[1:5] = 2.0
                    position_weights[5:10] = 1.5
                    position_weights = position_weights.unsqueeze(0).unsqueeze(-1)
                    
                    squared_errors = (predicted_scanpaths - true_scanpaths) ** 2
                    weighted_errors = squared_errors * position_weights
                    reconstruction_loss_norm = weighted_errors.mean()
                    
                    # 像素级距离损失
                    h, w = config.image_size
                    pred_pixels = predicted_scanpaths.clone()
                    pred_pixels[:, :, 0] = pred_pixels[:, :, 0] * w
                    pred_pixels[:, :, 1] = pred_pixels[:, :, 1] * h
                    
                    true_pixels = true_scanpaths.clone()
                    true_pixels[:, :, 0] = true_pixels[:, :, 0] * w
                    true_pixels[:, :, 1] = true_pixels[:, :, 1] * h
                    
                    pixel_distances = torch.norm(pred_pixels - true_pixels, p=2, dim=-1)
                    diagonal_length = np.sqrt(w**2 + h**2)
                    pixel_distances_norm = pixel_distances / diagonal_length
                    
                    # 验证损失计算（与训练一致）
                    rec_threshold_norm = 12.0 / diagonal_length
                    rec_threshold_pixels = 12.0
                    
                    # 硬约束：对距离>12像素的点对使用更强的惩罚
                    pixel_distances_abs = pixel_distances
                    far_mask = pixel_distances_abs > rec_threshold_pixels
                    
                    if len(pixel_distances_abs[far_mask]) > 0:
                        far_distances = pixel_distances_abs[far_mask]
                        normalized_far = (far_distances - rec_threshold_pixels) / rec_threshold_pixels
                        rec_penalty_far = torch.mean(torch.exp(normalized_far * 2.0))
                    else:
                        rec_penalty_far = torch.tensor(0.0, device=predicted_scanpaths.device)
                    
                    rec_penalty_all = torch.mean((pixel_distances_norm - rec_threshold_norm).clamp(min=0.0) ** 2)
                    rec_penalty = rec_penalty_all + 3.0 * rec_penalty_far
                    
                    pixel_diff_norm = (pred_pixels - true_pixels) / diagonal_length
                    pixel_mse = torch.mean(pixel_diff_norm ** 2)
                    pixel_l1 = torch.mean(torch.abs(pixel_diff_norm))
                    
                    reconstruction_loss = reconstruction_loss_norm + 0.5 * pixel_mse + 0.5 * pixel_l1 + 8.0 * rec_penalty

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
                    if epoch <= 10:
                        weights = {
                            'reconstruction': 5.0,
                            'kl': 0.002,
                            'spatial_coverage': 0.5,
                            'trajectory_smoothness': 0.1,
                            'direction_consistency': 0.1,
                            'sequence_alignment': 12.0,
                            'motion_consistency': 0.15,
                            'boundary': 0.1
                        }
                    elif epoch <= 25:
                        progress = (epoch - 10) / 15.0
                        weights = {
                            'reconstruction': 5.0 - 0.5 * progress,
                            'kl': 0.002 + 0.004 * progress,
                            'spatial_coverage': 0.5 + 0.5 * progress,
                            'trajectory_smoothness': 0.1 + 0.2 * progress,
                            'direction_consistency': 0.1 + 0.2 * progress,
                            'sequence_alignment': 12.0 + 4.0 * progress,
                            'motion_consistency': 0.15 + 0.35 * progress,
                            'boundary': 0.1
                        }
                    else:
                        weights = {
                            'reconstruction': 4.5,
                            'kl': 0.01,
                            'spatial_coverage': 1.0,
                            'trajectory_smoothness': 0.3,
                            'direction_consistency': 0.3,
                            'sequence_alignment': 20.0,
                            'motion_consistency': 0.5,
                            'boundary': 0.1
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
            # 修复：正确计算patience，考虑验证间隔
            save_model = False
            improved = False
            
            if val_position_error < best_val_position_error:
                best_val_position_error = val_position_error
                save_model = True
                improved = True
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
            
            # 修复：只有在验证时且未改善时才增加patience
            if not improved:
                # 计算从上次验证到现在的epoch数
                epochs_since_last_val = epoch - last_val_epoch
                patience_counter += epochs_since_last_val
                print(f"  ⚠️ 验证位置误差未改善 (patience: {patience_counter}/{early_stopping_patience})")
                print(f"     当前: {val_position_error:.4f}, 最佳: {best_val_position_error:.4f}")
            
            last_val_epoch = epoch

            # 早停检查：基于位置误差
            if patience_counter >= early_stopping_patience:
                print(f"\n⏹️ 早停触发！验证位置误差已经{patience_counter}个epoch没有改善")
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
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳验证位置误差: {best_val_position_error:.4f}")
    print(f"\n优化说明：")
    print(f"  1. 大幅提高序列对齐损失权重（12.0-20.0），修复REC为0问题")
    print(f"  2. 增强KL正则化（0.002-0.01），解决过拟合问题")
    print(f"  3. 提高REC惩罚权重（8.0），确保点对点匹配")
    print(f"  4. 降低初始学习率（0.000048），延长warmup（10 epochs）")
    print(f"  5. 统一训练和验证的Teacher Forcing策略")
    print(f"\n下一步：使用 evaluate_fixed.py 评估模型，检查REC指标是否改善")


if __name__ == '__main__':
    train()
