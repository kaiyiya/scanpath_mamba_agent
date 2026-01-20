"""
Y方向注意力机制
用于鼓励Y方向（纬度）的多样性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class YDirectionAttention(nn.Module):
    """Y方向注意力机制，鼓励Y方向多样性"""

    def __init__(self, d_model=384):
        super().__init__()
        self.y_projection = nn.Linear(d_model, d_model // 2)
        self.y_attention = nn.MultiheadAttention(
            embed_dim=d_model // 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.y_mlp = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )

        # 位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2)
        )

    def forward(self, features, history_y_positions):
        """
        Args:
            features: (B, d_model) - 当前特征
            history_y_positions: (B, T) - 历史Y位置
        Returns:
            y_bias: (B, 1) - Y方向偏置
        """
        B = features.size(0)

        # 投影特征
        y_features = self.y_projection(features).unsqueeze(1)  # (B, 1, d_model//2)

        # 编码历史位置
        history_embed = self.pos_encoder(history_y_positions.unsqueeze(-1))  # (B, T, d_model//2)

        # 注意力：query当前，key/value历史
        attended, _ = self.y_attention(y_features, history_embed, history_embed)

        # 生成Y方向偏置
        y_bias = self.y_mlp(attended.squeeze(1))  # (B, 1)

        return y_bias
