"""多尺度视觉聚合模块（从 inference_deploy/msufold/models/mst_net.py 移植）。

Attention-based Multi-Scale Visual Aggregation：
对视觉 backbone 各层 hidden states 做均匀采样（低/中/高层），
用注意力动态加权融合，替代只取最后一层特征的做法。
"""

import torch
from torch import nn


class MultiScaleVisualAdapter(nn.Module):
    """
    Attention-based Multi-Scale Visual Aggregation
    Dynamically weights features from different Transformer layers (Low/Mid/High level).
    """

    def __init__(self, dim=768, num_scales=4):
        super().__init__()
        self.num_scales = num_scales

        # Attention weight predictor
        # Input: [B, N, D], Output: [B, N, 1] (logits)
        self.attention_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

        # Final projection
        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, all_hidden_states):
        """
        Args:
            all_hidden_states: List of hidden states from all layers [L, B, N, D]
        Returns:
            fused_features: [B, N, D]
        """
        # 1. Uniformly sample layers
        num_layers = len(all_hidden_states)
        indices = [int(i * num_layers / self.num_scales) for i in range(self.num_scales)]
        selected = [all_hidden_states[i] for i in indices]

        # 2. Stack features: [B, N, K, D] where K=num_scales
        stacked_features = torch.stack(selected, dim=2)

        # 3. Compute Attention Weights
        # [B, N, K, D] -> [B, N, K, 1]
        attn_logits = self.attention_net(stacked_features)
        attn_weights = torch.softmax(attn_logits, dim=2)

        # 4. Weighted Sum
        weighted_features = (stacked_features * attn_weights).sum(dim=2)

        # 5. Final Projection
        fused_features = self.fusion(weighted_features)

        return fused_features
