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
        self.position_encoder = nn.Sequential(
            nn.Linear(2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model)
        )
        
        # ==================== 特征融合 ====================
        # 门控融合：融合全局和局部特征
        self.gated_fusion = SimpleGatedFusion(dim=d_model)
        
        # 特征更新：用局部特征更新全局特征
        self.feature_update_conv = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3, stride=1, padding=1,
            groups=d_model,
            bias=False
        )
        self.feature_update_norm = nn.LayerNorm(d_model)
        self.feature_update_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # 特征融合：Mamba状态 + 注意力特征 + 局部特征 + 位置
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
        # 预测隐变量的均值和方差
        self.latent_mu = nn.Linear(d_model, d_model // 2)
        self.latent_logvar = nn.Linear(d_model, d_model // 2)
        
        # 从隐变量解码位置
        # 改进：使用更好的初始化策略，让输出更容易覆盖全范围
        self.position_decoder = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(d_model // 4, 2),
            nn.Tanh()
        )
        
        # 改进初始化：让位置解码器输出更容易覆盖全范围
        # 初始化最后一层，让输出更容易分散
        nn.init.uniform_(self.position_decoder[-2].weight, -0.5, 0.5)
        nn.init.constant_(self.position_decoder[-2].bias, 0.0)  # 初始化为0，Tanh后为0（转换后为0.5）
        
        # 改进：初始化logvar层，让初始方差更大，增加多样性
        nn.init.constant_(self.latent_logvar.bias, -1.0)  # 初始logvar=-1，对应std≈0.6，增加初始多样性
    
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
        
        Args:
            images: 原始图像 (B, 3, H, W)
            positions: 归一化位置 [0, 1] (B, 2)
        Returns:
            patches: cropped patches (B, 3, patch_size, patch_size)
            feat_grid: 用于特征更新的grid
        """
        B = positions.size(0)
        H, W = self.image_size
        patch_size = self.focus_patch_size
        
        theta = torch.zeros((B, 2, 3), device=images.device)
        
        # 计算patch坐标
        patch_coordinate = positions * torch.tensor([H, W], device=images.device, dtype=torch.float32)
        patch_coordinate = patch_coordinate - patch_size / 2  # 中心对齐
        
        # 分别clamp X和Y坐标
        x_center = torch.clamp(patch_coordinate[:, 0], 0, H - patch_size)
        y_center = torch.clamp(patch_coordinate[:, 1], 0, W - patch_size)
        x1, x2 = x_center, x_center + patch_size
        y1, y2 = y_center, y_center + patch_size
        
        # 构建仿射变换矩阵
        theta[:, 0, 0] = patch_size / W
        theta[:, 1, 1] = patch_size / H
        theta[:, 0, 2] = -1 + (y1 + y2) / W
        theta[:, 1, 2] = -1 + (x1 + x2) / H
        
        # 生成grid并采样
        grid = F.affine_grid(theta.float(), torch.Size((B, 3, patch_size, patch_size)), align_corners=False)
        patches = F.grid_sample(images, grid, mode='bilinear', align_corners=False)
        
        # 生成特征更新用的grid
        feat_grid = F.affine_grid(theta.float(), torch.Size((B, 1, self.feature_size, self.feature_size)), align_corners=False)
        
        return patches, feat_grid
    
    def update_global_features(self, global_features, local_features, feat_grid):
        """
        用局部特征更新全局特征
        
        Args:
            global_features: 全局特征 (B, L, C)
            local_features: 局部特征 (B, L, C)
            feat_grid: 采样grid (B, feature_size, feature_size, 2)
        Returns:
            updated_features: 更新后的全局特征 (B, L, C)
        """
        B, L, C = global_features.shape
        H = W = self.feature_size
        
        # 归一化
        global_features_norm = self.feature_update_norm(global_features)
        
        # 转换为2D
        global_feat_2d = global_features_norm.permute(0, 2, 1).reshape(B, C, H, W)
        global_feat_2d = self.feature_update_conv(global_feat_2d)
        local_feat_2d = local_features.permute(0, 2, 1).reshape(B, C, H, W)
        
        # 计算融合权重
        fusion_weight = self.feature_update_mlp(local_features)  # (B, L, 1)
        fusion_weight = fusion_weight.expand(-1, -1, C)  # (B, L, C)
        
        # 使用grid_sample将局部特征采样到全局位置
        sampled_local = F.grid_sample(
            local_feat_2d,
            feat_grid,
            mode='bilinear',
            align_corners=False
        )  # (B, C, H, W)
        
        sampled_local = sampled_local.permute(0, 2, 3, 1).reshape(B, L, C)
        
        # 加权更新
        updated_features = global_features + fusion_weight * sampled_local
        
        return updated_features
    
    def forward(self, images, global_features, seq_len, gt_positions=None, teacher_forcing_ratio=0.5, 
                temperature=1.0, enable_early_stop=True, stop_threshold=0.5, min_steps=5):
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
        
        # 初始位置
        if use_teacher_forcing:
            prev_pos = gt_positions[:, 0, :].clone()  # 训练时使用真实起始点
        else:
            # 改进：从边缘区域随机选择初始位置，而不是从中心
            # 这更符合真实扫描路径通常从图像边缘开始的特点
            # 在图像的四个象限边缘附近随机选择
            # 方案：随机选择边缘区域（0-0.2 或 0.8-1.0）
            edge_choices = torch.rand(B, 2, device=device)
            # 随机选择是在左边缘还是右边缘（x方向）
            x_mask = edge_choices[:, 0] < 0.5
            x_positions = torch.where(x_mask, 
                                     torch.rand(B, device=device) * 0.2,  # 左边缘 [0, 0.2]
                                     torch.rand(B, device=device) * 0.2 + 0.8)  # 右边缘 [0.8, 1.0]
            # 随机选择是在上边缘还是下边缘（y方向）
            y_mask = edge_choices[:, 1] < 0.5
            y_positions = torch.where(y_mask,
                                     torch.rand(B, device=device) * 0.2,  # 上边缘 [0, 0.2]
                                     torch.rand(B, device=device) * 0.2 + 0.8)  # 下边缘 [0.8, 1.0]
            
            prev_pos = torch.stack([x_positions, y_positions], dim=1)  # (B, 2)
        
        positions = []
        updated_features = global_features
        stop_probs = []
        actual_lengths = torch.ones(B, dtype=torch.long, device=device) * seq_len  # 默认全部生成seq_len步
        
        # 序列生成循环（支持提前停止）
        for t in range(seq_len):
            # 1. 根据当前位置crop局部patch
            patches, feat_grid = self.get_img_patches(images, prev_pos)
            
            # 2. Focus网络提取局部高分辨率特征
            local_features = self.focus_net(patches)  # (B, N, C)
            
            # 3. 更新全局特征
            updated_features = self.update_global_features(
                updated_features, local_features, feat_grid
            )
            
            # 4. 融合全局和局部特征
            fused_features = self.gated_fusion(
                local_features.mean(dim=1),  # (B, C)
                updated_features.mean(dim=1)  # (B, C)
            ).unsqueeze(1)  # (B, 1, C)
            
            # 5. 编码位置
            pos_encoded = self.position_encoder(prev_pos).unsqueeze(1)  # (B, 1, C)
            
            # 6. 空间注意力
            attended_features, _ = self.spatial_attention(
                h_t,
                updated_features,
                updated_features
            )
            
            # 7. 特征融合：Mamba状态 + 注意力特征 + 局部特征 + 位置
            fused = torch.cat([h_t, attended_features, fused_features, pos_encoded], dim=-1)
            fused = self.feature_fusion(fused)
            
            # 8. Mamba序列建模
            h_t = self.mamba(fused)
            
            # 8.5. 自动停止判断（在位置预测之前）
            if enable_early_stop and not self.training:
                # 推理时：判断是否应该停止
                continue_prob = self.stop_classifier(h_t.squeeze(1))  # (B, 1)
                stop_prob = 1.0 - continue_prob  # (B, 1)
                stop_probs.append(stop_prob)
                
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
                    # 保存当前步的位置和VAE参数
                    positions.append(pos_t)
                    mus.append(mu)
                    logvars.append(logvar)
                    break
            elif enable_early_stop and self.training:
                # 训练时：也计算停止概率（用于损失），但不实际停止
                continue_prob = self.stop_classifier(h_t.squeeze(1))
                stop_prob = 1.0 - continue_prob
                stop_probs.append(stop_prob)
            
            # 9. VAE位置预测：概率建模防止过拟合
            combined_features = h_t.squeeze(1) + fused_features.squeeze(1)  # (B, d_model)
            
            # 预测隐变量的均值和方差
            mu = self.latent_mu(combined_features)  # (B, d_model//2)
            logvar = self.latent_logvar(combined_features)  # (B, d_model//2)
            
            # 重参数化采样（推理时增加温度以提高多样性）
            z = self.reparameterize(mu, logvar, temperature=temperature)  # (B, d_model//2)
            
            # 从隐变量解码位置
            pos_t = self.position_decoder(z)  # (B, 2), 范围[-1, 1]
            pos_t = (pos_t + 1.0) / 2.0  # 归一化到[0, 1]
            pos_t = torch.clamp(pos_t, 0.0, 1.0)
            
            # 保存mu和logvar用于计算KL散度
            if t == 0:
                mus = [mu]
                logvars = [logvar]
            else:
                mus.append(mu)
                logvars.append(logvar)
            
            positions.append(pos_t)
            
            # Teacher Forcing
            if use_teacher_forcing and t < seq_len - 1:
                if torch.rand(1).item() < teacher_forcing_ratio:
                    prev_pos = gt_positions[:, t, :]
                else:
                    prev_pos = pos_t.detach()
            else:
                prev_pos = pos_t
        
        # 处理变长序列：如果提前停止，需要padding到相同长度
        max_actual_len = actual_lengths.max().item()
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
                    stop_probs.append(stop_probs[-1] if stop_probs else torch.zeros(B, 1, device=device))
        
        positions = torch.stack(positions, dim=1)  # (B, seq_len, 2)
        mus = torch.stack(mus, dim=1)  # (B, seq_len, d_model//2)
        logvars = torch.stack(logvars, dim=1)  # (B, seq_len, d_model//2)
        
        if stop_probs:
            stop_probs = torch.stack(stop_probs, dim=1)  # (B, actual_len, 1)
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
                enable_early_stop=None, stop_threshold=0.5, min_steps=5):
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
            min_steps=min_steps
        )
        
        predicted_scanpaths, _, mus, logvars, stop_probs, actual_lengths = result
        
        if enable_early_stop:
            return predicted_scanpaths, mus, logvars, stop_probs, actual_lengths
        else:
            return predicted_scanpaths, mus, logvars
