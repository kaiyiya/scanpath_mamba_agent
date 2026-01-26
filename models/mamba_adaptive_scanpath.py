"""
Mamba-Adaptive扫描路径模型

架构说明：
----------
结合Mamba状态空间模型和AdaptiveNN的Focus机制，实现长期依赖的扫描路径预测。

核心组件：
1. MambaAdaptiveScanpathGenerator: 序列生成器，结合Focus机制和Mamba序列建模
2. MambaAdaptiveScanpath: 完整模型，包含Glance特征提取和序列生成

核心创新：
1. Mamba序列建模：长期依赖，参数共享（O(1)参数量）
2. 双流特征提取：Glance(全局) + Focus(局部高分辨率)
3. 动态特征更新：根据注视点更新全局表示
4. VAE概率建模：防止过拟合，增加预测多样性

工作流程：
----------
1. Glance网络提取全局特征 (B, 196, 384)
2. 对每个时间步：
   - 根据当前位置crop局部patch
   - Focus网络提取局部高分辨率特征
   - 更新全局特征表示
   - Mamba序列建模生成下一位置
   - VAE概率建模输出预测位置
3. 输出完整的扫描路径序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .improved_model_v5 import OptimizedSphereGlanceNet, OptimizedFocusNet, SimpleGatedFusion
from .y_attention import YDirectionAttention
from .feature_update_v2 import CrossAttentionFeatureUpdate

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Warning: mamba_ssm not installed. Please install: pip install mamba-ssm")
    Mamba = None


class MambaAdaptiveScanpathGenerator(nn.Module):
    """
    Mamba序列生成器 + AdaptiveNN的Focus机制
    
    在每个时间步执行：
    1. 根据当前位置crop局部patch
    2. Focus网络提取局部高分辨率特征
    3. 用局部特征更新全局特征表示
    4. 使用Mamba生成下一个位置
    
    Args:
        d_model: 模型维度，默认384
        d_state: Mamba SSM状态维度，默认256
        d_conv: Mamba局部卷积宽度，默认4
        expand: Mamba块扩展因子，默认2
        focus_patch_size: Focus网络处理的patch大小，默认224
        image_size: 输入图像尺寸，默认(256, 512)
        feature_size: 特征图大小，默认14
    """

    def __init__(self, d_model=384, d_state=256, d_conv=4, expand=2,
                 focus_patch_size=224, image_size=(256, 512), feature_size=14):
        super().__init__()

        if Mamba is None:
            raise ImportError("mamba_ssm not installed. Please install: pip install mamba-ssm")

        self.d_model = d_model
        self.focus_patch_size = focus_patch_size
        self.image_size = image_size
        self.feature_size = feature_size

        # ==================== Focus网络 ====================
        # 提取局部高分辨率特征
        self.focus_net = OptimizedFocusNet(
            input_size=focus_patch_size,
            output_dim=d_model,
            feature_size=feature_size
        )

        # ==================== Mamba序列建模 ====================
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # ==================== 位置编码 ====================
        # 关键改进：考虑360图像的周期性
        # 对于360图像，x坐标（经度）是周期性的，应该使用周期性编码
        # 方案：使用sin/cos编码来捕获周期性
        self.position_encoder = nn.Sequential(
            # 先对位置进行周期性编码
            # x坐标（经度）：使用sin/cos编码捕获周期性 [0, 1] -> [sin(2πx), cos(2πx)]
            # y坐标（纬度）：直接使用（不是周期性的）
            nn.Linear(4, d_model // 2),  # 输入：sin(2πx), cos(2πx), y, 1（bias）
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model)
        )

        # ==================== Y方向注意力 ====================
        # 鼓励Y方向多样性
        self.y_attention = YDirectionAttention(d_model=d_model)

        # ==================== 特征融合 ====================
        # 门控融合：融合全局和局部特征
        self.gated_fusion = SimpleGatedFusion(dim=d_model)

        # 特征更新：使用简化的Cross-Attention机制
        self.feature_updater = CrossAttentionFeatureUpdate(
            d_model=d_model,
            feature_size=feature_size
        )

        # 空间注意力
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )

        # 特征融合：Mamba状态 + 注意力特征 + 局部特征 + 位置
        # 注意：这个模块目前未使用，保留以备将来使用
        # 实际使用的是简单的相加和门控融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        # ==================== 自动停止机制 ====================
        # 停止分类器：判断是否应该继续生成
        self.stop_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # 输出继续概率 [0, 1]
        )

        # ==================== VAE概率建模 ====================
        # 改进：增强位置信息融合，使用位置特征直接参与解码
        # 预测隐变量的均值和方差（融合位置信息和历史位置信息）
        self.latent_mu = nn.Sequential(
            nn.Linear(d_model + 4, d_model),  # 加入当前位置和历史位置信息
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2)
        )
        self.latent_logvar = nn.Sequential(
            nn.Linear(d_model + 4, d_model),  # 加入当前位置和历史位置信息
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, d_model // 2)
        )

        # 从隐变量解码绝对位置（回退到原始方案）
        # 输出范围：线性输出，然后用Sigmoid映射到[0, 1]
        self.position_decoder = nn.Sequential(
            nn.Linear(d_model // 2 + 4, d_model // 2),  # 加入当前位置和历史位置信息
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 4, d_model // 8),
            nn.LayerNorm(d_model // 8),
            nn.GELU(),
            nn.Linear(d_model // 8, 2)  # 线性输出，后续用Sigmoid
        )

        # 标准初始化（不需要特殊初始化）
        # Sigmoid会自动将输出映射到[0, 1]

        # 改进：初始化logvar层，使用更合理的初始方差（增加初始多样性）
        # 注意：latent_logvar是Sequential，需要初始化最后一层的bias
        if hasattr(self.latent_logvar[-1], 'bias') and self.latent_logvar[-1].bias is not None:
            # 增加初始logvar，对应更大的初始std，增加多样性
            nn.init.constant_(self.latent_logvar[-1].bias, -1.0)  # 初始logvar=-1.0，对应std≈0.61，增加初始多样性

    def reparameterize(self, mu, logvar, temperature=1.0):
        """
        VAE重参数化技巧
        z = mu + eps * sigma * temperature, 其中eps ~ N(0, 1)
        
        Args:
            mu: 均值 (B, d_model//2)
            logvar: 对数方差 (B, d_model//2)
            temperature: 采样温度，>1时增加多样性，<1时减少多样性
        Returns:
            z: 采样后的隐变量 (B, d_model//2)
        """
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_img_patches(self, images, positions):
        """
        根据注视点位置crop图像patch

        关键改进：处理360图像的边界连续性
        - 360图像是等距圆柱投影（Equirectangular Projection）
        - 左右边界是连续的（x=0和x=W是同一个位置）
        - 当patch跨越左右边界时，需要wrap around处理

        Args:
            images: 原始图像 (B, 3, H, W)，360度等距圆柱投影图像
            positions: 归一化位置 [0, 1] (B, 2)，(y, x) = (纬度归一化, 经度归一化)
        Returns:
            patches: cropped patches (B, 3, patch_size, patch_size)
            feat_grid: 用于特征更新的grid
        """
        B = positions.size(0)
        H, W = self.image_size
        patch_size = self.focus_patch_size

        patches_list = []
        feat_grids_list = []

        for i in range(B):
            # 在计算patch时进行严格裁剪，确保位置在有效范围内
            # positions格式: (x, y) 其中 x是经度[0,1], y是纬度[0,1]
            pos_x = torch.clamp(positions[i, 0], 0.0, 1.0)  # 经度
            pos_y = torch.clamp(positions[i, 1], 0.0, 1.0)  # 纬度

            # 计算patch中心坐标（像素）
            x_center = pos_x * W - patch_size / 2  # X坐标（经度）
            y_center = pos_y * H - patch_size / 2  # Y坐标（纬度）

            # Y坐标clamp（纬度不是周期性的）
            y_start = int(torch.clamp(y_center, 0, H - patch_size).item())

            # X坐标wrap around（经度是周期性的）
            x_start = int(x_center.item()) % W

            # 提取patch
            if x_start + patch_size <= W:
                # 不跨越边界
                patch = images[i:i+1, :, y_start:y_start+patch_size, x_start:x_start+patch_size]
            else:
                # 跨越右边界，需要wrap around
                right_part = images[i:i+1, :, y_start:y_start+patch_size, x_start:]
                left_part = images[i:i+1, :, y_start:y_start+patch_size, :(x_start+patch_size-W)]
                patch = torch.cat([right_part, left_part], dim=3)

            patches_list.append(patch)

            # 生成feat_grid（用于特征更新）
            theta = torch.zeros((1, 2, 3), device=images.device)
            theta[:, 0, 0] = patch_size / W
            theta[:, 1, 1] = patch_size / H
            theta[:, 0, 2] = -1 + (x_start + x_start + patch_size) / W
            theta[:, 1, 2] = -1 + (y_start + y_start + patch_size) / H
            feat_grid = F.affine_grid(theta, torch.Size((1, 1, self.feature_size, self.feature_size)),
                                       align_corners=False)
            feat_grids_list.append(feat_grid)

        patches = torch.cat(patches_list, dim=0)
        feat_grids = torch.cat(feat_grids_list, dim=0)

        return patches, feat_grids

    def update_global_features(self, global_features, local_features, feat_grid):
        """
        用局部特征更新全局特征
        
        关键逻辑说明：
        1. local_features是局部patch的特征，形状(B, L, C)，其中L=feature_size*feature_size
        2. local_features的空间布局是局部patch的空间布局（14x14）
        3. feat_grid用于将局部特征的空间位置映射到全局特征图的空间位置
        4. 使用grid_sample将局部特征采样到全局位置，然后用于更新全局特征
        
        Args:
            global_features: 全局特征 (B, L, C)，其中L=feature_size*feature_size=196
            local_features: 局部特征 (B, L, C)，从局部patch提取的特征
            feat_grid: 采样grid (B, feature_size, feature_size, 2)，用于空间位置映射
        Returns:
            updated_features: 更新后的全局特征 (B, L, C)
        """
        B, L, C = global_features.shape
        H = W = self.feature_size
        assert L == H * W, f"特征长度L={L}应该等于H*W={H*W}"

        # 归一化全局特征
        global_features_norm = self.feature_update_norm(global_features)

        # 转换为2D格式：(B, L, C) -> (B, C, H, W)
        global_feat_2d = global_features_norm.permute(0, 2, 1).reshape(B, C, H, W)
        # 对全局特征进行深度卷积（空间卷积，不改变通道数）
        global_feat_2d = self.feature_update_conv(global_feat_2d)
        
        # 将局部特征转换为2D格式：(B, L, C) -> (B, C, H, W)
        # 注意：local_features的空间布局是局部patch的空间布局（14x14）
        local_feat_2d = local_features.permute(0, 2, 1).reshape(B, C, H, W)

        # 计算融合权重（基于局部特征的重要性）
        fusion_weight = self.feature_update_mlp(local_features)  # (B, L, 1)
        fusion_weight = fusion_weight.expand(-1, -1, C)  # (B, L, C)

        # 使用grid_sample将局部特征采样到全局位置
        # feat_grid的坐标系统：[-1, 1]，用于将局部特征的空间位置映射到全局特征图的空间位置
        # 注意：feat_grid是从get_img_patches生成的，它基于图像patch的位置
        # 这里假设局部特征的空间布局和全局特征图的空间布局都是14x14
        sampled_local = F.grid_sample(
            local_feat_2d,
            feat_grid,
            mode='bilinear',
            align_corners=False
        )  # (B, C, H, W)

        # 转换回序列格式：(B, C, H, W) -> (B, L, C)
        sampled_local = sampled_local.permute(0, 2, 3, 1).reshape(B, L, C)

        # 加权更新：用局部特征更新全局特征
        # fusion_weight控制更新的强度，基于局部特征的重要性
        updated_features = global_features + fusion_weight * sampled_local

        return updated_features

    def forward(self, images, global_features, seq_len, gt_positions=None, teacher_forcing_ratio=0.5,
                temperature=1.0, enable_early_stop=True, stop_threshold=0.5, min_steps=5, use_gt_start=True):
        """
        前向传播
        
        Args:
            images: 原始图像 (B, 3, H, W)
            global_features: 全局特征 (B, N, C)
            seq_len: 最大序列长度（实际可能提前停止）
            gt_positions: 真实位置（训练时使用） (B, seq_len, 2)
            teacher_forcing_ratio: Teacher Forcing比例
            temperature: 采样温度
            enable_early_stop: 是否启用自动停止（推理时建议True，训练时建议False）
            stop_threshold: 停止阈值，继续概率低于此值时停止（默认0.5）
            min_steps: 最小步数，至少生成这么多步才允许停止（默认5）
        Returns:
            positions: 预测位置序列 (B, actual_len, 2)，实际长度可能小于seq_len
            updated_features: 最终更新后的全局特征 (B, N, C)
            mus: VAE均值 (B, actual_len, d_model//2)
            logvars: VAE对数方差 (B, actual_len, d_model//2)
            stop_probs: 停止概率序列 (B, actual_len, 1)，用于分析
            actual_lengths: 每个样本的实际长度 (B,)，用于处理变长序列
        """
        B, N, C = global_features.shape
        device = images.device

        # 初始化序列状态：使用加权平均，保留空间信息
        feature_importance = torch.norm(global_features, p=2, dim=-1, keepdim=True)  # (B, N, 1)
        feature_weights = F.softmax(feature_importance, dim=1)  # (B, N, 1)
        h_t = (global_features * feature_weights).sum(dim=1, keepdim=True)  # (B, 1, C)
        # 添加少量噪声以增加多样性
        h_t = h_t + torch.randn_like(h_t) * 0.02

        # 判断是否使用Teacher Forcing
        use_teacher_forcing = gt_positions is not None and self.training

        # 初始位置（改进：训练时使用真实起始点）
        if use_gt_start and gt_positions is not None:
            # 训练/验证时：使用真实起始点，确保序列对齐
            prev_pos = gt_positions[:, 0, :].clone()
        elif use_teacher_forcing and torch.rand(1).item() < teacher_forcing_ratio:
            # 训练时：按Teacher Forcing比例使用真实起始点
            prev_pos = gt_positions[:, 0, :].clone()
        else:
            # 改进：使用更合理的初始位置分布
            # 关键修复：增加初始位置的多样性，避免所有样本都从相似位置开始
            # 方案1：从整个图像范围随机选择（更符合真实分布）
            # 方案2：从边缘区域随机选择（更符合扫描路径通常从边缘开始的特点）
            # 使用方案1和方案2的混合：60%从边缘，40%从全图（调整比例，增加全图比例）
            use_edge = torch.rand(B, device=device) < 0.6

            # 边缘位置：从四个边缘区域随机选择（扩大边缘区域范围）
            edge_choices = torch.rand(B, 2, device=device)
            x_mask = edge_choices[:, 0] < 0.5
            x_positions_edge = torch.where(x_mask,
                                           torch.rand(B, device=device) * 0.25,  # 左边缘 [0, 0.25]（扩大范围）
                                           torch.rand(B, device=device) * 0.25 + 0.75)  # 右边缘 [0.75, 1.0]（扩大范围）
            y_mask = edge_choices[:, 1] < 0.5
            y_positions_edge = torch.where(y_mask,
                                           torch.rand(B, device=device) * 0.25,  # 上边缘 [0, 0.25]（扩大范围）
                                           torch.rand(B, device=device) * 0.25 + 0.75)  # 下边缘 [0.75, 1.0]（扩大范围）

            # 全图位置：从整个图像范围随机选择
            x_positions_full = torch.rand(B, device=device)  # [0, 1]
            y_positions_full = torch.rand(B, device=device)  # [0, 1]

            # 混合：60%边缘，40%全图
            x_positions = torch.where(use_edge, x_positions_edge, x_positions_full)
            y_positions = torch.where(use_edge, y_positions_edge, y_positions_full)

            prev_pos = torch.stack([x_positions, y_positions], dim=1)  # (B, 2)
            # 放宽约束范围，允许更接近边界的初始位置
            prev_pos = torch.clamp(prev_pos, 0.02, 0.98)  # 约束在[0.02, 0.98]范围内，增加初始位置多样性

        positions = []
        updated_features = global_features
        stop_probs = []
        actual_lengths = torch.ones(B, dtype=torch.long, device=device) * seq_len  # 默认全部生成seq_len步

        # 改进：累积历史特征序列，让Mamba看到完整的历史信息
        # 关键问题：之前每次只给Mamba一个时间步的特征，无法有效建模序列依赖
        # 改进：累积历史特征，让Mamba看到完整的序列上下文
        history_features = []  # 累积历史特征序列

        # 序列生成循环（支持提前停止）
        for t in range(seq_len):
            # 1. 根据当前位置crop局部patch
            patches, feat_grid = self.get_img_patches(images, prev_pos)

            # 2. Focus网络提取局部高分辨率特征
            local_features = self.focus_net(patches)  # (B, L, C)，其中L=feature_size*feature_size=196

            # 3. 更新全局特征（使用简化的Cross-Attention机制）
            updated_features = self.feature_updater(
                updated_features, local_features, prev_pos
            )

            # 4. 融合全局和局部特征（改进：使用更强大的融合）
            local_context = local_features.mean(dim=1)  # (B, C)
            global_context = updated_features.mean(dim=1)  # (B, C)
            fused_features = self.gated_fusion(local_context, global_context)  # (B, C)

            # 5. 编码位置（改进：考虑360图像的周期性）
            # 关键修复：对于360图像，x坐标（经度）是周期性的
            # 使用sin/cos编码来捕获周期性：x -> [sin(2πx), cos(2πx)]
            # 这样模型可以学习到x=0和x=1是相邻的
            pos_x_periodic = torch.stack([
                torch.sin(2 * torch.pi * prev_pos[:, 0]),  # sin(2πx)
                torch.cos(2 * torch.pi * prev_pos[:, 0])   # cos(2πx)
            ], dim=-1)  # (B, 2)
            pos_y = prev_pos[:, 1:2]  # (B, 1)，纬度不是周期性的
            # 组合：sin(2πx), cos(2πx), y
            pos_encoded_input = torch.cat([pos_x_periodic, pos_y], dim=-1)  # (B, 3)
            # 添加一个常数特征（bias），让模型学习绝对位置
            pos_encoded_input = torch.cat([pos_encoded_input, torch.ones(B, 1, device=device)], dim=-1)  # (B, 4)
            pos_encoded = self.position_encoder(pos_encoded_input)  # (B, C)

            # 6. 空间注意力（改进：使用累积的历史状态）
            if t == 0:
                # 第一步：使用初始状态
                attended_features, _ = self.spatial_attention(
                    h_t.squeeze(1).unsqueeze(1),  # (B, 1, C)
                    updated_features,
                    updated_features
                )
            else:
                # 后续步骤：使用历史状态的平均
                # 修复：history_features是列表，每个元素是(B, C)，应该用dim=0来stack
                recent_history = history_features[-min(5, t):]
                history_context = torch.stack(recent_history, dim=0)  # (T, B, C)
                history_context = history_context.transpose(0, 1)  # (B, T, C)
                history_context = history_context.mean(dim=1, keepdim=True)  # (B, 1, C)
                attended_features, _ = self.spatial_attention(
                    history_context,
                    updated_features,
                    updated_features
                )

            # 7. 特征融合：融合当前时间步的所有信息
            # 改进：使用残差连接和更强大的融合
            current_feature = fused_features + pos_encoded + attended_features.squeeze(1)  # (B, C)
            current_feature = current_feature.unsqueeze(1)  # (B, 1, C)

            # 8. 累积历史特征
            history_features.append(current_feature.squeeze(1))  # (B, C)

            # 9. Mamba序列建模（改进：使用累积的历史序列）
            # 关键改进：让Mamba看到完整的历史序列，而不是只看到当前时间步
            if t == 0:
                # 第一步：只有一个时间步
                history_seq = current_feature  # (B, 1, C)
            else:
                # 后续步骤：累积历史序列（使用最近的历史，避免过长）
                # 修复：history_features是列表，每个元素是(B, C)，应该用dim=0来stack
                recent_history = history_features[-min(10, t + 1):]  # 最多使用最近10步
                history_seq = torch.stack(recent_history, dim=0)  # (T, B, C)
                history_seq = history_seq.transpose(0, 1)  # (B, T, C)

            # Mamba处理序列（关键：现在Mamba可以看到历史序列）
            h_t = self.mamba(history_seq)  # (B, T, C) -> (B, T, C)
            h_t = h_t[:, -1:, :]  # 只取最后一个时间步的输出 (B, 1, C)

            # 10. VAE位置预测：概率建模防止过拟合（改进：融合位置信息和历史信息）
            # 改进：融合Mamba输出、当前特征和历史位置信息
            mamba_output = h_t.squeeze(1)  # (B, d_model)
            current_context = fused_features  # (B, d_model)

            # 融合历史位置信息（改进：让模型看到之前的预测位置）
            if t > 0:
                # 使用最近3个位置的平均作为历史位置信息
                # 修复：positions是列表，每个元素是(B, 2)，应该用dim=0来stack
                recent_positions = torch.stack(positions[-min(3, t):], dim=0)  # (T, B, 2)
                recent_positions = recent_positions.transpose(0, 1)  # (B, T, 2)
                history_pos = recent_positions.mean(dim=1)  # (B, 2)
            else:
                history_pos = prev_pos  # 第一步：使用当前位置

            # 融合所有信息（改进：使用更强大的融合策略）
            # 关键修复：简单相加可能导致信息丢失，使用加权融合
            # 问题：mamba_output和current_context直接相加，可能导致某些信息被稀释
            # 修复：使用更稳定的门控机制，避免sigmoid饱和
            # 改进：使用归一化的点积，避免数值不稳定
            mamba_norm = F.normalize(mamba_output, p=2, dim=-1, eps=1e-8)
            context_norm = F.normalize(current_context, p=2, dim=-1, eps=1e-8)
            similarity = (mamba_norm * context_norm).sum(dim=-1, keepdim=True)  # (B, 1), 范围[-1, 1]
            # 将相似度从[-1, 1]映射到[0, 1]，使用更稳定的方式
            fusion_weight = (similarity + 1.0) / 2.0  # (B, 1), 范围[0, 1]
            # 使用加权融合，但保留残差连接
            combined_features = fusion_weight * mamba_output + (1 - fusion_weight) * current_context  # (B, d_model)
            features_with_pos = torch.cat([combined_features, prev_pos, history_pos], dim=-1)  # (B, d_model+4)

            # 预测隐变量的均值和方差（使用融合位置信息的特征）
            mu = self.latent_mu(features_with_pos)  # (B, d_model//2)
            logvar = self.latent_logvar(features_with_pos)  # (B, d_model//2)

            # 改进：动态调整logvar，确保有足够的多样性
            # 关键修复：放宽logvar的下限，允许更大的多样性
            # 问题：之前min=-3.0对应std≈0.05，太小，导致采样过于集中
            # 修复：降低下限到-2.0，对应std≈0.37，增加多样性
            logvar = torch.clamp(logvar, min=-2.0, max=2.0)  # 限制logvar范围，确保std在[0.37, 2.7]

            # 方案2：降低VAE随机性，改善序列对齐
            # 训练时也减少随机性，使用更确定性的采样
            if self.training:
                # 训练时：使用较低的temperature，减少随机性
                # temperature=0.5意味着std减半，更接近确定性
                z = self.reparameterize(mu, logvar, temperature=0.3)  # (B, d_model//2)
            else:
                # 推理时：直接使用mu，完全确定性
                z = mu  # (B, d_model//2)

            # ==================== 方案B：预测相对位移而不是绝对位置 ====================
            # 回退到绝对位置预测（相对位移预测导致LEV恶化）
            # 从隐变量解码绝对位置（融合当前位置和历史位置信息）
            z_with_pos = torch.cat([z, prev_pos, history_pos], dim=-1)  # (B, d_model//2+4)
            pos_t = self.position_decoder(z_with_pos)  # (B, 2), 范围[-1, 1]

            # Sigmoid将输出映射到[0, 1]范围
            pos_t = torch.sigmoid(pos_t)  # (B, 2)

            # ==================== Y方向注意力偏置 ====================
            # 在位置预测后，添加Y方向偏置以增加Y方向多样性
            if t > 0 and hasattr(self, 'y_attention'):
                # 收集历史Y位置
                history_y = torch.stack([p[:, 1] for p in positions[-min(5, t):]], dim=1)  # (B, T)

                # 计算Y方向偏置
                y_bias = self.y_attention(combined_features, history_y)  # (B, 1)

                # 添加偏置到Y坐标（非inplace操作，避免破坏梯度图）
                y_coord = pos_t[:, 1] + 0.1 * y_bias.squeeze(1)
                # 裁剪到有效范围
                y_coord = torch.clamp(y_coord, 0.0, 1.0)
                # 重新组合坐标
                pos_t = torch.stack([pos_t[:, 0], y_coord], dim=-1)

            # 处理360图像的边界连续性
            # X坐标：wrap around处理（360度连续性）
            pos_t_x = pos_t[:, 0] % 1.0  # 映射到[0, 1]

            # Y坐标：裁剪到有效范围
            pos_t_y = torch.clamp(pos_t[:, 1], 0.0, 1.0)

            pos_t = torch.stack([pos_t_x, pos_t_y], dim=-1)  # (B, 2)

            # 保存mu和logvar用于计算KL散度
            if t == 0:
                mus = [mu]
                logvars = [logvar]
            else:
                mus.append(mu)
                logvars.append(logvar)

            positions.append(pos_t)

            # 8.5. 自动停止判断（在位置预测之后，确保所有变量都已定义）
            if enable_early_stop:
                # 计算停止概率
                continue_prob = self.stop_classifier(h_t.squeeze(1))  # (B, 1)
                stop_prob = 1.0 - continue_prob  # (B, 1)
                stop_probs.append(stop_prob)

                # 推理时：判断是否应该停止
                if not self.training:
                    # 判断哪些样本应该停止（至少生成min_steps步）
                    should_stop = (stop_prob.squeeze(1) > (1.0 - stop_threshold)) & (t >= min_steps - 1)  # (B,)

                    # 更新实际长度（记录停止的步数）
                    actual_lengths = torch.where(
                        (actual_lengths == seq_len) & should_stop,
                        torch.tensor(t + 1, device=device),
                        actual_lengths
                    )

                    # 如果所有样本都停止了，提前退出循环
                    if should_stop.all():
                        break
                # 训练时：只计算停止概率（用于损失），但不实际停止

            # Teacher Forcing（修复：相对位移模式下应该使用预测位置）
            # 关键修复：由于现在预测的是相对位移，Teacher Forcing不应该直接替换prev_pos
            # 而应该让模型使用自己预测的位置，只在计算损失时与GT对比
            # 这样才能保持相对位移的连续性
            if use_teacher_forcing and t < seq_len - 1:
                # 始终使用预测的位置作为下一步的输入
                # Teacher Forcing的作用通过损失函数体现，而不是直接替换位置
                prev_pos = pos_t.detach()
            else:
                prev_pos = pos_t

        # 处理变长序列：如果提前停止，需要padding到相同长度
        actual_len = len(positions)

        # 如果提前停止，需要padding
        if actual_len < seq_len:
            # 用最后一个位置填充
            last_pos = positions[-1]  # (B, 2)
            last_mu = mus[-1]  # (B, d_model//2)
            last_logvar = logvars[-1]  # (B, d_model//2)

            for _ in range(seq_len - actual_len):
                positions.append(last_pos)
                mus.append(last_mu)
                logvars.append(last_logvar)
                if stop_probs:
                    # 如果stop_probs不为空，使用最后一个stop_prob填充
                    stop_probs.append(stop_probs[-1])
                elif enable_early_stop:
                    # 如果启用early_stop但stop_probs为空（不应该发生），创建零tensor
                    stop_probs.append(torch.zeros(B, 1, device=device))

        # 修复：positions、mus、logvars是列表，每个元素是(B, ...)，应该用dim=0来stack
        positions = torch.stack(positions, dim=0)  # (seq_len, B, 2)
        positions = positions.transpose(0, 1)  # (B, seq_len, 2)

        mus = torch.stack(mus, dim=0)  # (seq_len, B, d_model//2)
        mus = mus.transpose(0, 1)  # (B, seq_len, d_model//2)

        logvars = torch.stack(logvars, dim=0)  # (seq_len, B, d_model//2)
        logvars = logvars.transpose(0, 1)  # (B, seq_len, d_model//2)

        if stop_probs:
            stop_probs = torch.stack(stop_probs, dim=0)  # (actual_len, B, 1)
            stop_probs = stop_probs.transpose(0, 1)  # (B, actual_len, 1)
            # Padding stop_probs到seq_len
            if stop_probs.size(1) < seq_len:
                padding = stop_probs[:, -1:].repeat(1, seq_len - stop_probs.size(1), 1)
                stop_probs = torch.cat([stop_probs, padding], dim=1)
        else:
            stop_probs = torch.zeros(B, seq_len, 1, device=device)

        return positions, updated_features, mus, logvars, stop_probs, actual_lengths


class MambaAdaptiveScanpath(nn.Module):
    """
    完整的Mamba-Adaptive扫描路径模型
    
    结合Mamba状态空间模型和AdaptiveNN的Focus机制，实现长期依赖的扫描路径预测。
    
    Args:
        config: 配置对象，包含模型超参数
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.feature_dim = config.feature_dim

        # Glance网络：全局特征提取
        self.glance_net = OptimizedSphereGlanceNet(
            input_size=config.image_size,
            output_dim=config.feature_dim,
            feature_size=config.feature_size
        )

        # Mamba-Adaptive生成器
        self.mamba_adaptive_generator = MambaAdaptiveScanpathGenerator(
            d_model=config.feature_dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            focus_patch_size=getattr(config, 'focus_patch_size', 224),
            image_size=config.image_size,
            feature_size=config.feature_size
        )

        # LayerNorm
        self.norm = nn.LayerNorm(config.feature_dim)

    def forward(self, images, gt_scanpaths=None, teacher_forcing_ratio=0.5, temperature=1.0,
                enable_early_stop=None, stop_threshold=0.5, min_steps=5, use_gt_start=True):
        """
        前向传播

        Args:
            images: 输入图像 (B, 3, H, W)
            gt_scanpaths: 真实扫描路径（训练时使用） (B, seq_len, 2)
            teacher_forcing_ratio: Teacher Forcing比例
            temperature: 采样温度，训练时用1.0，推理时可>1.0以增加多样性
            enable_early_stop: 是否启用自动停止（None时自动判断：训练=False，推理=True）
            stop_threshold: 停止阈值（默认0.5）
            min_steps: 最小步数（默认5）
            use_gt_start: 是否使用真实起始点（默认True，改善LEV指标）
        Returns:
            predicted_scanpaths: 预测的扫描路径 (B, seq_len, 2)
            mus: VAE均值 (B, seq_len, d_model//2)
            logvars: VAE对数方差 (B, seq_len, d_model//2)
            stop_probs: 停止概率序列 (B, seq_len, 1)，如果启用early_stop
            actual_lengths: 实际长度 (B,)，如果启用early_stop
        """
        # 自动判断是否启用early_stop
        if enable_early_stop is None:
            enable_early_stop = not self.training  # 训练时False，推理时True

        # 1. 提取全局特征
        global_features = self.glance_net(images)  # (B, 196, 384)
        global_features = self.norm(global_features)

        # 2. Mamba-Adaptive序列生成（带Focus和特征更新）+ VAE + 自动停止
        result = self.mamba_adaptive_generator(
            images,
            global_features,
            self.seq_len,
            gt_positions=gt_scanpaths,
            teacher_forcing_ratio=teacher_forcing_ratio,
            temperature=temperature,
            enable_early_stop=enable_early_stop,
            stop_threshold=stop_threshold,
            min_steps=min_steps,
            use_gt_start=use_gt_start
        )

        predicted_scanpaths, _, mus, logvars, stop_probs, actual_lengths = result

        if enable_early_stop:
            return predicted_scanpaths, mus, logvars, stop_probs, actual_lengths
        else:
            return predicted_scanpaths, mus, logvars
