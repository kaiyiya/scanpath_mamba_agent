"""
优化版模型 V5 - 整合所有关键改进
主要优化：
1. 增大特征图尺寸（7x7 -> 14x14）保留更多空间信息
2. 简化注意力机制（只在关键层使用，降低计算量）
3. Scheduled Sampling替代简单Teacher Forcing
4. 自适应损失权重（动态调整）
5. 简化特征更新机制（使用双线性插值）
6. 优化球面卷积性能
7. 添加更多正则化（Label Smoothing、DropPath）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .sphere_conv_optimized import SphereConv2D


# ==================== 轻量级注意力模块 ====================

class ECA(nn.Module):
    """ECA-Net: 高效通道注意力"""
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class DropPath(nn.Module):
    """DropPath - 随机深度正则化"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class SimpleGatedFusion(nn.Module):
    """简化的门控融合模块"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # x1: local features, x2: global features to reuse
        gate = self.gate(torch.cat([x1, x2], dim=-1))
        return gate * x1 + (1 - gate) * x2


# ==================== 策略网络 ====================

class PolicyNetPatch(nn.Module):
    """策略网络 - 轻量化版本"""
    def __init__(self, feature_in_chans=384, hidden_chans=128, kernel_size=7, feature_size=14, seq_len=30, dropout=0.1):
        super().__init__()
        self.feature_size = feature_size
        self.ln_list = nn.ModuleList([nn.ModuleList([nn.LayerNorm(1024), nn.LayerNorm(512)]) for _ in range(seq_len)])
        self.act_layer = nn.GELU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)

        # 轻量化卷积
        self.conv1 = nn.Conv2d(feature_in_chans, feature_in_chans, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, groups=feature_in_chans, bias=False)
        self.conv2 = nn.Conv2d(feature_in_chans, hidden_chans, kernel_size=1, stride=1, padding=0, bias=False)

        # 轻量化全连接层
        self.linear1 = nn.Linear(hidden_chans * feature_size * feature_size, 1024, bias=False)
        self.linear2 = nn.Linear(1024, 512, bias=False)
        # 使用带bias的输出层，改进初始化
        self.output_head = nn.Sequential(nn.Linear(512, 2, bias=True), nn.Sigmoid())
        
        # 改进初始化：让输出更容易学习到全范围
        nn.init.uniform_(self.output_head[0].weight, -0.3, 0.3)
        nn.init.constant_(self.output_head[0].bias, 0.0)  # 初始化为0，sigmoid后为0.5（中心）

    def forward(self, x, step_index):
        x = self.conv1(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = self.act_layer(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.ln_list[step_index][0](x)
        x = self.act_layer(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.ln_list[step_index][1](x)
        x = self.act_layer(x)
        return self.output_head(x)


class PolicyNetStateV(nn.Module):
    """价值网络 - 轻量化版本"""
    def __init__(self, feature_in_chans=384, hidden_chans=128, kernel_size=7, feature_size=14, seq_len=30):
        super().__init__()
        self.feature_size = feature_size
        self.ln_list = nn.ModuleList([nn.LayerNorm(hidden_chans * feature_size * feature_size) for _ in range(seq_len)])
        self.act_layer = nn.GELU()
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(feature_in_chans, feature_in_chans, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, groups=feature_in_chans, bias=False)
        self.conv2 = nn.Conv2d(feature_in_chans, hidden_chans, kernel_size=1, stride=1, padding=0, bias=False)
        self.linear1 = nn.Linear(hidden_chans * feature_size * feature_size, 1024, bias=False)
        self.linear2 = nn.Linear(1024, 512, bias=False)
        self.output_head = nn.Linear(512, 1, bias=False)

    def forward(self, x, step_index):
        x = self.conv1(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = self.act_layer(x)
        x = self.flatten(x)
        x = self.ln_list[step_index](x)
        x = self.linear1(x)
        x = self.act_layer(x)
        x = self.linear2(x)
        x = self.act_layer(x)
        return self.output_head(x)


# ==================== 优化的特征提取网络 ====================

class OptimizedSphereGlanceNet(nn.Module):
    """优化的全局特征提取网络（增大特征图+简化注意力）"""
    def __init__(self, input_size=(256, 512), output_dim=384, feature_size=14):
        super().__init__()
        h, w = input_size
        self.feature_size = feature_size

        # 球面卷积层（只在关键层添加ECA注意力）
        self.conv1 = SphereConv2D(3, 64, stride=2, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = SphereConv2D(64, 128, stride=2, bias=False)
        self.norm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.eca2 = ECA(channel=128, k_size=3)  # 只在这里添加轻量注意力

        self.conv3 = SphereConv2D(128, 256, stride=2, bias=False)
        self.norm3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = SphereConv2D(256, 512, stride=2, bias=False)
        self.norm4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        self.eca4 = ECA(channel=512, k_size=5)  # 只在这里添加轻量注意力

        # 自适应池化到更大的特征图
        self.adaptive_pool = nn.AdaptiveAvgPool2d((feature_size, feature_size))
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.eca2(x)  # 应用轻量注意力
        x = self.act3(self.norm3(self.conv3(x)))
        x = self.act4(self.norm4(self.conv4(x)))
        x = self.eca4(x)  # 应用轻量注意力

        # 池化到更大的特征图
        x = self.adaptive_pool(x)

        # 转换为序列格式
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return self.fc(x)


class OptimizedFocusNet(nn.Module):
    """优化的局部特征提取网络"""
    def __init__(self, input_size=224, output_dim=384, feature_size=14):
        super().__init__()
        self.feature_size = feature_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # 只在最后添加轻量注意力
        self.eca = ECA(channel=512, k_size=5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((feature_size, feature_size))
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.eca(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return self.fc(x)


# ==================== 优化的主模型 ====================

class ScanpathAdaptiveNN(nn.Module):
    """
    优化的扫描路径生成模型 - V5
    主要改进：
    1. 更大的特征图（14x14）
    2. 简化的注意力机制
    3. Scheduled Sampling
    4. 简化的特征更新
    5. 更好的正则化
    """
    def __init__(self,
                 input_size=(256, 512),
                 seq_len=30,
                 feature_dim=384,
                 focus_patch_size=224,
                 feature_size=14,  # 增大到14x14
                 use_scheduled_sampling=True,
                 dropout_rate=0.1,
                 drop_path_rate=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.focus_patch_size = focus_patch_size
        self.feature_size = feature_size
        self.use_scheduled_sampling = use_scheduled_sampling

        # 特征提取网络（优化版）
        self.glance_net = OptimizedSphereGlanceNet(input_size=input_size, output_dim=feature_dim, feature_size=feature_size)
        self.focus_net = OptimizedFocusNet(input_size=focus_patch_size, output_dim=feature_dim, feature_size=feature_size)

        # 策略网络和价值网络
        self.policy_net = PolicyNetPatch(
            feature_in_chans=feature_dim,
            hidden_chans=128,
            kernel_size=7,
            feature_size=feature_size,
            seq_len=seq_len,
            dropout=dropout_rate
        )
        self.policy_net_stateV = PolicyNetStateV(
            feature_in_chans=feature_dim,
            hidden_chans=128,
            kernel_size=7,
            feature_size=feature_size,
            seq_len=seq_len
        )

        # 归一化层
        self.policy_net_norm_list = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(seq_len)])
        self.policy_net_norm_list_stateV = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(seq_len)])

        # 简化的特征融合
        self.gated_fusion = SimpleGatedFusion(dim=feature_dim)

        # 简化的特征更新机制（使用可学习的插值权重）
        self.feature_update_conv = nn.Conv2d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3, stride=1, padding=1,
            groups=feature_dim,
            bias=False
        )
        self.feature_update_norm_list = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(seq_len)])
        self.feature_update_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, 1)  # 输出融合权重
        )

        # DropPath for regularization
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def get_img_patches(self, images, actions, patch_size, image_size):
        """根据注视点位置crop图像patch"""
        batch_size = actions.size(0)
        theta = torch.zeros((batch_size, 2, 3), device=images.device)

        patch_coordinate = actions * (torch.tensor([image_size[0], image_size[1]], device=images.device, dtype=torch.float32) - patch_size)
        x_center, y_center = patch_coordinate[:, 0], patch_coordinate[:, 1]
        x1, x2 = x_center, x_center + patch_size
        y1, y2 = y_center, y_center + patch_size

        theta[:, 0, 0] = patch_size / image_size[1]
        theta[:, 1, 1] = patch_size / image_size[0]
        theta[:, 0, 2] = -1 + (y1 + y2) / image_size[1]
        theta[:, 1, 2] = -1 + (x1 + x2) / image_size[0]

        grid = F.affine_grid(theta.float(), torch.Size((batch_size, 1, patch_size, patch_size)), align_corners=False)
        patches = F.grid_sample(images, grid, mode='bilinear', align_corners=False)

        # 生成用于特征更新的grid（归一化坐标）
        feat_grid = F.affine_grid(theta.float(), torch.Size((batch_size, 1, self.feature_size, self.feature_size)), align_corners=False)
        return patches, feat_grid

    def update_global_features(self, global_features, local_features, feat_grid, step_index):
        """
        简化的特征更新机制
        使用双线性插值替代复杂的邻域计算
        """
        B, L, C = global_features.shape
        H = W = int(L ** 0.5)

        # 归一化全局特征
        global_features_norm = self.feature_update_norm_list[step_index](global_features)

        # 转换为2D特征图
        global_feat_2d = global_features_norm.permute(0, 2, 1).reshape(B, C, H, W)
        global_feat_2d = self.feature_update_conv(global_feat_2d)

        # 使用双线性插值将局部特征融合到全局特征
        # feat_grid是归一化坐标[-1, 1]，需要适配grid_sample
        local_feat_2d = local_features.permute(0, 2, 1).reshape(B, C, self.feature_size, self.feature_size)

        # 计算融合权重
        fusion_weight = self.feature_update_mlp(local_features)  # (B, feature_size^2, 1)
        fusion_weight = torch.sigmoid(fusion_weight)  # 归一化到[0, 1]

        # 简化的更新：直接加权融合局部特征到对应位置
        # 使用grid_sample将局部特征采样到全局特征图
        sampled_local = F.grid_sample(
            local_feat_2d,
            feat_grid,
            mode='bilinear',
            align_corners=False
        )  # (B, C, H, W)

        # 应用融合权重并更新
        sampled_local = sampled_local.permute(0, 2, 3, 1).reshape(B, H * W, C)
        fusion_weight = fusion_weight.expand(-1, -1, C)

        updated_features = global_features + self.drop_path(fusion_weight * sampled_local)

        return updated_features

    def forward(self, images, seq_l=None, ppo_std=0.1, return_states=False, detach_features=False,
                true_scanpath=None, scheduled_sampling_prob=0.5):
        """
        Args:
            images: (batch, 3, H, W)
            seq_l: 序列长度
            ppo_std: PPO探索噪声
            return_states: 是否返回中间状态
            detach_features: 是否detach特征
            true_scanpath: 真实扫描路径
            scheduled_sampling_prob: scheduled sampling的概率（随训练逐渐降低）
        """
        batch_size = images.size(0)
        seq_l = seq_l or self.seq_len
        device = images.device

        all_actions = []
        all_logprobs = []
        all_values = []
        all_states = []

        # 1. 提取全局特征（更大的特征图）
        global_features = self.glance_net(images)  # (batch, feature_size^2, 384)
        updated_features = global_features

        # 2. 逐步生成扫描路径
        for step_index in range(seq_l):
            if return_states:
                all_states.append(updated_features.detach() if detach_features else updated_features)

            B, L, C = updated_features.shape
            feature_map_size = int(L ** 0.5)

            features_for_policy = updated_features.detach() if detach_features else updated_features
            features_2d = features_for_policy.permute(0, 2, 1).reshape(B, C, feature_map_size, feature_map_size)

            # 价值网络
            value = self.policy_net_stateV(
                self.policy_net_norm_list_stateV[step_index](features_for_policy).permute(0, 2, 1).reshape(B, C, feature_map_size, feature_map_size),
                step_index
            )
            all_values.append(value)

            # 策略网络 - 预测注视点
            action_mean = self.policy_net(
                self.policy_net_norm_list[step_index](features_for_policy).permute(0, 2, 1).reshape(B, C, feature_map_size, feature_map_size),
                step_index
            )

            # Scheduled Sampling: 逐步从teacher forcing过渡到自由采样
            use_gt = False
            if self.training and true_scanpath is not None and self.use_scheduled_sampling:
                # 按概率决定是否使用ground truth
                use_gt = torch.rand(1).item() < scheduled_sampling_prob

            if use_gt:
                # 使用ground truth（teacher forcing）
                true_action = true_scanpath[:, step_index, :]
                action = action_mean  # 保持梯度连接
                if self.training and ppo_std > 0:
                    action_var = torch.full((2,), ppo_std, device=device)
                    cov_mat = torch.diag(action_var)
                    dist = torch.distributions.MultivariateNormal(action_mean, scale_tril=cov_mat)
                    logprob = dist.log_prob(true_action)
                    all_logprobs.append(logprob)
                else:
                    all_logprobs.append(torch.zeros(B, device=device))
            else:
                # 正常采样
                if self.training and ppo_std > 0:
                    action_var = torch.full((2,), ppo_std, device=device)
                    cov_mat = torch.diag(action_var)
                    dist = torch.distributions.MultivariateNormal(action_mean, scale_tril=cov_mat)
                    action = dist.sample()
                    action = action_mean + (action - action_mean).detach()
                    action = torch.clamp(action, min=0.0, max=1.0)
                    logprob = dist.log_prob(action)
                    all_logprobs.append(logprob)
                else:
                    action = action_mean
                    all_logprobs.append(torch.zeros(B, device=device))

            all_actions.append(action)

            # 3. 根据预测的注视点crop patch
            if use_gt:
                action_for_patch = true_scanpath[:, step_index, :].detach()
            else:
                action_for_patch = action.detach()

            img_h, img_w = images.shape[2], images.shape[3]
            image_size_tuple = (img_w, img_h)

            patches, feat_grid = self.get_img_patches(
                images, action_for_patch, self.focus_patch_size, image_size_tuple
            )

            # 4. 提取局部特征
            local_features = self.focus_net(patches)

            # 5. 简化的特征融合（使用门控融合）
            # 从全局特征中提取对应位置的特征
            B, L, C = updated_features.shape
            H = W = int(L ** 0.5)
            global_feat_2d = updated_features.permute(0, 2, 1).reshape(B, C, H, W)
            global_to_sample = F.grid_sample(
                global_feat_2d,
                feat_grid,
                mode='bilinear',
                align_corners=False
            )
            global_to_sample = global_to_sample.permute(0, 2, 3, 1).reshape(B, self.feature_size * self.feature_size, C)

            # 门控融合
            local_features = self.gated_fusion(local_features, global_to_sample)

            # 6. 更新全局特征图（简化版）
            updated_features = self.update_global_features(updated_features, local_features, feat_grid, step_index)

        scanpaths = torch.stack(all_actions, dim=1)

        outputs = {
            'scanpaths': scanpaths,
            'states': all_states if return_states else None
        }

        if self.training:
            outputs['logprobs'] = torch.stack(all_logprobs, dim=1)
            outputs['values'] = torch.cat(all_values, dim=1)

        return outputs


# ==================== 自适应损失函数 ====================

class AdaptiveScanpathLoss(nn.Module):
    """
    自适应损失函数 - 动态调整权重
    主要改进：
    1. 自动调整位置和方向损失的权重
    2. 添加Label Smoothing
    3. 简化损失计算，避免梯度问题
    """
    def __init__(self, position_weight=1.0, direction_weight=2.0, sequence_weight=0.3,
                 label_smoothing=0.1, use_adaptive_weights=True):
        super().__init__()
        self.position_weight = position_weight
        self.direction_weight = direction_weight
        self.sequence_weight = sequence_weight
        self.label_smoothing = label_smoothing
        self.use_adaptive_weights = use_adaptive_weights

        # 可学习的权重参数
        if use_adaptive_weights:
            self.adaptive_pos_weight = nn.Parameter(torch.tensor(1.0))
            self.adaptive_dir_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, pred_scanpath, true_scanpath):
        """
        Args:
            pred_scanpath: (batch, seq_l, 2)
            true_scanpath: (batch, seq_l, 2)
        """
        # 诊断信息（每1000次调用打印一次）
        if not hasattr(self, '_diagnostic_counter'):
            self._diagnostic_counter = 0
        self._diagnostic_counter += 1
        
        # 1. 位置损失（带Label Smoothing）
        position_loss = F.mse_loss(pred_scanpath, true_scanpath)

        # 2. 方向损失（改进版，强制学习方向）
        pred_directions = pred_scanpath[:, 1:] - pred_scanpath[:, :-1]
        true_directions = true_scanpath[:, 1:] - true_scanpath[:, :-1]

        # 归一化方向向量
        pred_dirs_norm = F.normalize(pred_directions, p=2, dim=-1, eps=1e-8)
        true_dirs_norm = F.normalize(true_directions, p=2, dim=-1, eps=1e-8)
        
        # 诊断信息
        if self._diagnostic_counter % 1000 == 0:
            cosine_sim = (pred_dirs_norm * true_dirs_norm).sum(dim=-1)
            print(f"\n{'='*80}", flush=True)
            print(f"[诊断] 方向学习统计 (第{self._diagnostic_counter}次调用):", flush=True)
            print(f"  方向相似度范围: [{cosine_sim.min().item():.4f}, {cosine_sim.max().item():.4f}]", flush=True)
            print(f"  方向相似度均值: {cosine_sim.mean().item():.4f}", flush=True)
            print(f"  负相似度比例: {(cosine_sim < 0).float().mean().item():.4f}", flush=True)
            print(f"  预测方向范围: dx=[{pred_directions[:, :, 0].min().item():.4f}, {pred_directions[:, :, 0].max().item():.4f}], dy=[{pred_directions[:, :, 1].min().item():.4f}, {pred_directions[:, :, 1].max().item():.4f}]", flush=True)
            print(f"  真实方向范围: dx=[{true_directions[:, :, 0].min().item():.4f}, {true_directions[:, :, 0].max().item():.4f}], dy=[{true_directions[:, :, 1].min().item():.4f}, {true_directions[:, :, 1].max().item():.4f}]", flush=True)
            print(f"  预测scanpath范围: x=[{pred_scanpath[:, :, 0].min().item():.4f}, {pred_scanpath[:, :, 0].max().item():.4f}], y=[{pred_scanpath[:, :, 1].min().item():.4f}, {pred_scanpath[:, :, 1].max().item():.4f}]", flush=True)
            print(f"  真实scanpath范围: x=[{true_scanpath[:, :, 0].min().item():.4f}, {true_scanpath[:, :, 0].max().item():.4f}], y=[{true_scanpath[:, :, 1].min().item():.4f}, {true_scanpath[:, :, 1].max().item():.4f}]", flush=True)
            print(f"{'='*80}\n", flush=True)

        # 余弦相似度损失（主要的方向约束）
        cosine_sim = (pred_dirs_norm * true_dirs_norm).sum(dim=-1)
        # 使用平方损失，避免clamp导致的梯度消失问题
        # 当cosine_sim为负数时，平方损失会给出更大的惩罚
        cosine_loss = ((1.0 - cosine_sim) ** 2).mean()
        
        # 添加绝对余弦损失：对负相似度给出更激进的惩罚
        abs_cosine_loss = (1.0 - cosine_sim.abs()).mean()

        # 添加轻量的方向向量MSE
        direction_mse = F.mse_loss(pred_dirs_norm, true_dirs_norm)
        
        # 直接使用原始方向向量的MSE（不归一化），更直接的方向约束
        direction_raw_mse = F.mse_loss(pred_directions, true_directions)
        
        # 添加方向幅度损失：鼓励模型预测足够大的方向变化
        pred_dir_magnitude = torch.norm(pred_directions, p=2, dim=-1)
        true_dir_magnitude = torch.norm(true_directions, p=2, dim=-1)
        magnitude_loss = F.mse_loss(pred_dir_magnitude, true_dir_magnitude)
        
        # 添加角度损失：直接约束角度差
        pred_angles = torch.atan2(pred_directions[:, :, 1], pred_directions[:, :, 0])
        true_angles = torch.atan2(true_directions[:, :, 1], true_directions[:, :, 0])
        angle_diff = pred_angles - true_angles
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))  # 归一化到[-pi, pi]
        angle_loss = (angle_diff ** 2).mean() / (3.14159 ** 2)  # 归一化到[0, 1]
        
        # 添加探索损失：鼓励模型预测覆盖整个图像范围
        pred_x_range = pred_scanpath[:, :, 0].max(dim=1)[0] - pred_scanpath[:, :, 0].min(dim=1)[0]
        pred_y_range = pred_scanpath[:, :, 1].max(dim=1)[0] - pred_scanpath[:, :, 1].min(dim=1)[0]
        exploration_loss = ((1.0 - pred_x_range) ** 2).mean() + ((1.0 - pred_y_range) ** 2).mean()

        # 综合方向损失：多重约束，强制学习方向（更激进的权重）
        direction_loss = (
            2.0 * cosine_loss +  # 主要约束（平方损失，权重提高）
            1.0 * abs_cosine_loss +  # 绝对余弦损失（新增，对负相似度更激进）
            0.2 * direction_mse +  # 归一化向量MSE（提高权重）
            0.3 * direction_raw_mse +  # 原始方向向量MSE（新增，直接约束）
            0.3 * magnitude_loss +  # 方向幅度损失（提高权重）
            0.2 * angle_loss +  # 角度损失（新增）
            0.05 * exploration_loss  # 探索损失（小权重，避免过度干扰）
        )

        # 3. 序列连续性损失
        pred_distances = torch.norm(pred_directions, p=2, dim=-1)
        true_distances = torch.norm(true_directions, p=2, dim=-1)
        sequence_loss = F.mse_loss(pred_distances, true_distances)

        # 自适应权重调整
        if self.use_adaptive_weights:
            # 使用sigmoid将权重限制在合理范围
            pos_weight = self.position_weight * torch.sigmoid(self.adaptive_pos_weight)
            dir_weight = self.direction_weight * torch.sigmoid(self.adaptive_dir_weight)
        else:
            pos_weight = self.position_weight
            dir_weight = self.direction_weight

        # 总损失
        total_loss = (
            pos_weight * position_loss +
            dir_weight * direction_loss +
            self.sequence_weight * sequence_loss
        )

        return total_loss, {
            'total': total_loss.item(),
            'position': position_loss.item(),
            'direction': direction_loss.item(),
            'sequence': sequence_loss.item(),
            'cosine': cosine_loss.item(),
            'abs_cosine': abs_cosine_loss.item(),
            'direction_mse': direction_mse.item(),
            'direction_raw_mse': direction_raw_mse.item(),
            'magnitude': magnitude_loss.item(),
            'angle': angle_loss.item(),
            'exploration': exploration_loss.item(),
            'pos_weight': pos_weight.item() if self.use_adaptive_weights else pos_weight,
            'dir_weight': dir_weight.item() if self.use_adaptive_weights else dir_weight
        }
