"""
先进的多模态融合分类器。

基于Transformer架构，使用交叉注意力机制融合点云和图像特征。
支持多种融合策略和分类头设计。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Tuple


class CrossModalAttention(nn.Module):
    """
    交叉模态注意力模块。
    
    允许点云特征和图像特征之间进行交互。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim必须能被num_heads整除"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: 查询张量 (B, N1, D)
            key: 键张量 (B, N2, D)
            value: 值张量 (B, N2, D)
        
        Returns:
            注意力输出 (B, N1, D)
        """
        B, N1, D = query.shape
        N2 = key.shape[1]
        
        # 投影
        q = self.q_proj(query).view(B, N1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N1, D_h)
        k = self.k_proj(key).view(B, N2, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N2, D_h)
        v = self.v_proj(value).view(B, N2, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N2, D_h)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N1, N2)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        out = torch.matmul(attn_weights, v)  # (B, H, N1, D_h)
        out = out.transpose(1, 2).contiguous().view(B, N1, D)  # (B, N1, D)
        out = self.out_proj(out)
        
        # 残差连接和层归一化
        out = self.norm(query + out)
        
        return out


class MultimodalFusionBlock(nn.Module):
    """
    多模态融合块。
    
    使用交叉注意力机制融合点云和图像特征。
    """

    def __init__(
        self,
        pointnext_dim: int,
        convnext_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 特征投影到统一维度
        self.pointnext_proj = nn.Linear(pointnext_dim, hidden_dim)
        self.convnext_proj = nn.Linear(convnext_dim, hidden_dim)
        
        # 交叉注意力模块
        self.pointnext_to_image = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.image_to_pointnext = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        pointnext_feat: torch.Tensor,
        convnext_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pointnext_feat: 点云特征 (B, D1) 或 (B, N, D1)
            convnext_feat: 图像特征 (B, D2) 或 (B, N, D2)
        
        Returns:
            融合后的点云特征和图像特征
        """
        # 如果输入是2D，添加序列维度
        if pointnext_feat.dim() == 2:
            pointnext_feat = pointnext_feat.unsqueeze(1)  # (B, 1, D1)
        if convnext_feat.dim() == 2:
            convnext_feat = convnext_feat.unsqueeze(1)  # (B, 1, D2)
        
        # 投影到统一维度
        pointnext_proj = self.pointnext_proj(pointnext_feat)  # (B, 1, hidden_dim)
        convnext_proj = self.convnext_proj(convnext_feat)  # (B, 1, hidden_dim)
        
        # 交叉注意力
        pointnext_fused = self.pointnext_to_image(
            pointnext_proj, convnext_proj, convnext_proj
        )  # (B, 1, hidden_dim)
        convnext_fused = self.image_to_pointnext(
            convnext_proj, pointnext_proj, pointnext_proj
        )  # (B, 1, hidden_dim)
        
        # 前馈网络
        pointnext_fused = pointnext_fused + self.ffn(pointnext_fused)
        pointnext_fused = self.ffn_norm(pointnext_fused)
        
        convnext_fused = convnext_fused + self.ffn(convnext_fused)
        convnext_fused = self.ffn_norm(convnext_fused)
        
        # 移除序列维度
        pointnext_fused = pointnext_fused.squeeze(1)  # (B, hidden_dim)
        convnext_fused = convnext_fused.squeeze(1)  # (B, hidden_dim)
        
        return pointnext_fused, convnext_fused


class MultimodalClassifier(nn.Module):
    """
    先进的多模态融合分类器。
    
    基于Transformer架构，使用交叉注意力机制融合点云和图像特征。
    """

    def __init__(
        self,
        pointnext_dim: int = 256,
        convnext_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 4,  # 4类分类：类别0, 1, 2, 3
        num_fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_mode: Literal["cross_attention", "concat", "add", "gated"] = "cross_attention",
    ):
        """
        Args:
            pointnext_dim: PointNeXt特征维度
            convnext_dim: ConvNeXt特征维度
            hidden_dim: 融合后的隐藏维度
            num_classes: 分类类别数
            num_fusion_layers: 融合层数
            num_heads: 注意力头数
            dropout: Dropout比率
            fusion_mode: 融合模式
                - "cross_attention": 交叉注意力融合（推荐）
                - "concat": 简单拼接
                - "add": 相加（需要维度相同）
                - "gated": 门控融合
        """
        super().__init__()
        self.fusion_mode = fusion_mode
        self.hidden_dim = hidden_dim
        
        if fusion_mode == "cross_attention":
            # 构建多层融合块
            self.fusion_blocks = nn.ModuleList([
                MultimodalFusionBlock(
                    pointnext_dim if i == 0 else hidden_dim,
                    convnext_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout,
                )
                for i in range(num_fusion_layers)
            ])
            
            # 如果第一层输入维度不同，需要投影
            if pointnext_dim != hidden_dim:
                self.pointnext_proj = nn.Linear(pointnext_dim, hidden_dim)
            else:
                self.pointnext_proj = nn.Identity()
            
            if convnext_dim != hidden_dim:
                self.convnext_proj = nn.Linear(convnext_dim, hidden_dim)
            else:
                self.convnext_proj = nn.Identity()
            
            # 最终融合特征维度
            final_dim = hidden_dim * 2  # 拼接两个模态的融合特征
        
        elif fusion_mode == "concat":
            # 简单拼接
            self.pointnext_proj = nn.Linear(pointnext_dim, hidden_dim)
            self.convnext_proj = nn.Linear(convnext_dim, hidden_dim)
            final_dim = hidden_dim * 2
        
        elif fusion_mode == "add":
            # 相加融合（需要维度相同）
            assert pointnext_dim == convnext_dim, "add模式需要两个模态维度相同"
            self.pointnext_proj = nn.Linear(pointnext_dim, hidden_dim)
            self.convnext_proj = nn.Linear(convnext_dim, hidden_dim)
            final_dim = hidden_dim
        
        elif fusion_mode == "gated":
            # 门控融合
            self.pointnext_proj = nn.Linear(pointnext_dim, hidden_dim)
            self.convnext_proj = nn.Linear(convnext_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            final_dim = hidden_dim
        
        else:
            raise ValueError(f"未知的融合模式: {fusion_mode}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        pointnext_feat: torch.Tensor,
        convnext_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pointnext_feat: 点云特征 (B, D1)
            convnext_feat: 图像特征 (B, D2)
        
        Returns:
            分类logits (B, num_classes)
        """
        # 调试信息：检查输入特征维度
        if hasattr(self, '_debug_first_call'):
            pass
        else:
            self._debug_first_call = True
            print(f"[调试] 输入特征维度: pointnext_feat={pointnext_feat.shape}, convnext_feat={convnext_feat.shape}")
            print(f"[调试] 期望维度: pointnext_dim={self.fusion_blocks[0].pointnext_proj.in_features if hasattr(self.fusion_blocks[0], 'pointnext_proj') else 'N/A'}, "
                  f"convnext_dim={self.fusion_blocks[0].convnext_proj.in_features if hasattr(self.fusion_blocks[0], 'convnext_proj') else 'N/A'}")
        
        if self.fusion_mode == "cross_attention":
            # 直接传入原始特征，让fusion_block的第一层进行投影
            pointnext_proj = pointnext_feat
            convnext_proj = convnext_feat
            
            # 多层融合
            for i, fusion_block in enumerate(self.fusion_blocks):
                if i == 0 and hasattr(self, '_debug_first_call'):
                    # 检查传入 fusion_block 的特征维度
                    print(f"[调试] 传入 fusion_block[{i}] 的特征维度: pointnext={pointnext_proj.shape}, convnext={convnext_proj.shape}")
                pointnext_proj, convnext_proj = fusion_block(pointnext_proj, convnext_proj)
            
            # 如果输出是3D (B, 1, hidden_dim)，需要squeeze成2D (B, hidden_dim)
            if pointnext_proj.dim() == 3:
                pointnext_proj = pointnext_proj.squeeze(1)
            if convnext_proj.dim() == 3:
                convnext_proj = convnext_proj.squeeze(1)
            
            # 拼接融合后的特征
            fused_feat = torch.cat([pointnext_proj, convnext_proj], dim=1)
        
        elif self.fusion_mode == "concat":
            pointnext_proj = self.pointnext_proj(pointnext_feat)
            convnext_proj = self.convnext_proj(convnext_feat)
            fused_feat = torch.cat([pointnext_proj, convnext_proj], dim=1)
        
        elif self.fusion_mode == "add":
            pointnext_proj = self.pointnext_proj(pointnext_feat)
            convnext_proj = self.convnext_proj(convnext_feat)
            fused_feat = pointnext_proj + convnext_proj
        
        elif self.fusion_mode == "gated":
            pointnext_proj = self.pointnext_proj(pointnext_feat)
            convnext_proj = self.convnext_proj(convnext_feat)
            concat_feat = torch.cat([pointnext_proj, convnext_proj], dim=1)
            gate = self.gate(concat_feat)
            fused_feat = gate * pointnext_proj + (1 - gate) * convnext_proj
        
        # 分类
        logits = self.classifier(fused_feat)
        
        return logits


def create_classifier(
    pointnext_dim: int = 256,
    convnext_dim: int = 768,
    num_classes: int = 4,  # 4类分类：类别0, 1, 2, 3
    **kwargs
) -> MultimodalClassifier:
    """
    创建分类器的便捷函数。
    
    Args:
        pointnext_dim: PointNeXt特征维度
        convnext_dim: ConvNeXt特征维度
        num_classes: 分类类别数
        **kwargs: 其他参数传递给MultimodalClassifier
    
    Returns:
        分类器模型
    """
    return MultimodalClassifier(
        pointnext_dim=pointnext_dim,
        convnext_dim=convnext_dim,
        num_classes=num_classes,
        **kwargs
    )

