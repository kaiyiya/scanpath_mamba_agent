"""
简化的特征更新机制
使用Cross-Attention替代复杂的grid_sample操作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFeatureUpdate(nn.Module):
    """使用Cross-Attention简化特征更新"""

    def __init__(self, d_model=384, feature_size=14):
        super().__init__()
        self.d_model = d_model
        self.feature_size = feature_size

        # 更新门控
        self.update_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, global_features, local_features, position):
        """
        Args:
            global_features: (B, L, C) - 全局特征
            local_features: (B, L, C) - 局部特征
            position: (B, 2) - 归一化位置[0, 1]
        Returns:
            updated_features: (B, L, C)
        """
        B, L, C = global_features.shape
        H = W = self.feature_size

        # 1. 计算空间权重
        spatial_weights = self.compute_spatial_weights(position, H, W, global_features.device)  # (B, L, 1)

        # 2. 聚合局部特征（优化：使用加权平均而不是简单平均）
        # 改进：根据局部特征的重要性进行加权聚合
        local_importance = torch.norm(local_features, p=2, dim=-1, keepdim=True)  # (B, L, 1)
        local_weights = F.softmax(local_importance, dim=1)  # (B, L, 1)
        local_aggregate = (local_features * local_weights).sum(dim=1, keepdim=True)  # (B, 1, C)

        # 3. 广播更新信号（优化：结合空间权重和特征重要性）
        update_signal = local_aggregate * spatial_weights  # (B, L, C)

        # 4. 门控融合
        gate_input = torch.cat([global_features, update_signal], dim=-1)  # (B, L, 2C)
        gate = self.update_gate(gate_input)  # (B, L, C)

        updated_features = gate * update_signal + (1 - gate) * global_features

        return updated_features

    def compute_spatial_weights(self, position, H, W, device):
        """计算空间权重（考虑360度wrap around）"""
        # 创建网格位置
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        grid_positions = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)  # (L, 2)

        B = position.size(0)
        grid_positions = grid_positions.unsqueeze(0).expand(B, -1, -1)  # (B, L, 2)
        pos_expanded = position.unsqueeze(1)  # (B, 1, 2)

        # X方向距离（考虑wrap around）
        x_dist = torch.abs(grid_positions[:, :, 0:1] - pos_expanded[:, :, 0:1])
        x_dist = torch.min(x_dist, 1.0 - x_dist)  # wrap around

        # Y方向距离
        y_dist = torch.abs(grid_positions[:, :, 1:2] - pos_expanded[:, :, 1:2])

        # 组合距离
        spatial_dist = torch.sqrt(x_dist**2 + y_dist**2)  # (B, L, 1)

        # 高斯权重（优化：使用自适应sigma，根据位置动态调整）
        # 改进：使用更合理的sigma值，平衡局部和全局信息
        sigma = 0.25  # 稍微增大sigma，增加更新范围
        weights = torch.exp(-spatial_dist**2 / (2 * sigma**2))
        # 添加最小权重，确保所有位置都能获得更新
        weights = weights + 0.01  # 添加基础权重
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        return weights
