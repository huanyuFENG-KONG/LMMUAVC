"""
轻量级多模态分类器 - 专为小数据集设计

支持三模态融合（图像、Livox、Lidar 360）：
- 显式缺失建模：使用缺失掩码标记每个模态是否存在
- 置信度加权：根据检测置信度对每个模态的贡献进行加权

针对过拟合问题，提供三种轻量级模型：
1. TinyFusion: 超轻量级MLP（~50K参数）
2. CompactTransformer: 压缩版Transformer（~500K参数）
3. EfficientFusion: 平衡性能和参数量（~2M参数）
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple


class TinyFusion(nn.Module):
    """
    超轻量级MLP融合模型（四模态：图像、Livox、Lidar 360、Radar）
    
    参数量：~100,000（hidden_dim=128）
    适用场景：数据量 < 3,000
    
    支持显式缺失建模、置信度加权和点云数量加权
    """
    def __init__(
        self,
        image_dim: int = 768,  # ConvNeXt特征维度
        livox_dim: int = 512,  # Livox PointNeXt特征维度
        lidar360_dim: int = 512,  # Lidar 360 PointNeXt特征维度
        radar_dim: int = 512,  # Radar PointNeXt特征维度
        num_classes: int = 4,
        hidden_dim: int = 128,  # 128 = 32*4，每个模态32维
        dropout: float = 0.5,
        use_missing_embeddings: bool = True,
        use_missing_masks: bool = True,
        use_weighting: bool = True,
        use_confidence_weighting: bool = True,
        use_density_weighting: bool = True,
    ):
        super().__init__()
        
        # 保存配置
        self.use_missing_embeddings = use_missing_embeddings
        self.use_missing_masks = use_missing_masks
        self.use_weighting = use_weighting
        self.use_confidence_weighting = use_confidence_weighting
        self.use_density_weighting = use_density_weighting
        
        # 特征投影（降维到统一维度）
        self.image_proj = nn.Linear(image_dim, hidden_dim // 4)
        self.livox_proj = nn.Linear(livox_dim, hidden_dim // 4)
        self.lidar360_proj = nn.Linear(lidar360_dim, hidden_dim // 4)
        self.radar_proj = nn.Linear(radar_dim, hidden_dim // 4)
        
        # 缺失嵌入（learnable embedding for missing modality）
        if use_missing_embeddings:
            self.image_missing_emb = nn.Parameter(torch.randn(hidden_dim // 4))
            self.livox_missing_emb = nn.Parameter(torch.randn(hidden_dim // 4))
            self.lidar360_missing_emb = nn.Parameter(torch.randn(hidden_dim // 4))
            self.radar_missing_emb = nn.Parameter(torch.randn(hidden_dim // 4))
        else:
            # 如果禁用缺失嵌入，使用零向量（但仍需注册为参数以保持兼容性）
            self.register_buffer('image_missing_emb', torch.zeros(hidden_dim // 4))
            self.register_buffer('livox_missing_emb', torch.zeros(hidden_dim // 4))
            self.register_buffer('lidar360_missing_emb', torch.zeros(hidden_dim // 4))
            self.register_buffer('radar_missing_emb', torch.zeros(hidden_dim // 4))
        
        # 置信度投影（将置信度映射到权重）
        if use_confidence_weighting and use_weighting:
            self.confidence_proj = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        else:
            self.confidence_proj = None
        
        # 点云数量投影（将点云数量映射到权重，用于平衡不同模态）
        if use_density_weighting and use_weighting:
            self.point_count_proj = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        else:
            self.point_count_proj = None
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 初始化缺失嵌入为小值（仅在启用且是Parameter时）
        if self.use_missing_embeddings and hasattr(self, 'image_missing_emb'):
            # 只初始化 Parameter 类型（Buffer 已经初始化为零，不需要重新初始化）
            if isinstance(self.image_missing_emb, nn.Parameter):
                nn.init.normal_(self.image_missing_emb, 0, 0.01)
                nn.init.normal_(self.livox_missing_emb, 0, 0.01)
                nn.init.normal_(self.lidar360_missing_emb, 0, 0.01)
                nn.init.normal_(self.radar_missing_emb, 0, 0.01)
    
    def forward(
        self,
        image_feat: torch.Tensor,  # [B, D_img]
        livox_feat: torch.Tensor,   # [B, D_livox]
        lidar360_feat: torch.Tensor, # [B, D_lidar360]
        radar_feat: torch.Tensor,  # [B, D_radar]
        image_mask: Optional[torch.Tensor] = None,  # [B] 1表示存在，0表示缺失
        livox_mask: Optional[torch.Tensor] = None,
        lidar360_mask: Optional[torch.Tensor] = None,
        radar_mask: Optional[torch.Tensor] = None,
        image_conf: Optional[torch.Tensor] = None,  # [B] 置信度 [0, 1]
        livox_conf: Optional[torch.Tensor] = None,
        lidar360_conf: Optional[torch.Tensor] = None,
        radar_conf: Optional[torch.Tensor] = None,
        image_point_count: Optional[torch.Tensor] = None,  # [B] 点云数量（用于权重计算）
        livox_point_count: Optional[torch.Tensor] = None,
        lidar360_point_count: Optional[torch.Tensor] = None,
        radar_point_count: Optional[torch.Tensor] = None,
    ):
        B = image_feat.shape[0]
        device = image_feat.device
        
        # 检测缺失（如果未提供mask，则通过特征范数判断）
        if image_mask is None:
            if self.use_missing_masks:
                image_mask = (image_feat.norm(dim=-1) > 1e-6).float()
            else:
                image_mask = torch.ones(B, device=device)  # 禁用掩码时，使用全1
        else:
            image_mask = image_mask.to(device)
        if livox_mask is None:
            if self.use_missing_masks:
                livox_mask = (livox_feat.norm(dim=-1) > 1e-6).float()
            else:
                livox_mask = torch.ones(B, device=device)
        else:
            livox_mask = livox_mask.to(device)
        if lidar360_mask is None:
            if self.use_missing_masks:
                lidar360_mask = (lidar360_feat.norm(dim=-1) > 1e-6).float()
            else:
                lidar360_mask = torch.ones(B, device=device)
        else:
            lidar360_mask = lidar360_mask.to(device)
        if radar_mask is None:
            if self.use_missing_masks:
                radar_mask = (radar_feat.norm(dim=-1) > 1e-6).float()
            else:
                radar_mask = torch.ones(B, device=device)
        else:
            radar_mask = radar_mask.to(device)
        
        # 默认置信度为1.0（如果未提供），否则确保在正确设备上
        if image_conf is None:
            image_conf = torch.ones(B, device=device)
        else:
            image_conf = image_conf.to(device)
        if livox_conf is None:
            livox_conf = torch.ones(B, device=device)
        else:
            livox_conf = livox_conf.to(device)
        if lidar360_conf is None:
            lidar360_conf = torch.ones(B, device=device)
        else:
            lidar360_conf = lidar360_conf.to(device)
        if radar_conf is None:
            radar_conf = torch.ones(B, device=device)
        else:
            radar_conf = radar_conf.to(device)
        
        # 默认点云数量为0（如果未提供），否则确保在正确设备上
        if image_point_count is None:
            image_point_count = torch.zeros(B, device=device)
        else:
            image_point_count = image_point_count.to(device)
        if livox_point_count is None:
            livox_point_count = torch.zeros(B, device=device)
        else:
            livox_point_count = livox_point_count.to(device)
        if lidar360_point_count is None:
            lidar360_point_count = torch.zeros(B, device=device)
        else:
            lidar360_point_count = lidar360_point_count.to(device)
        if radar_point_count is None:
            radar_point_count = torch.zeros(B, device=device)
        else:
            radar_point_count = radar_point_count.to(device)
        
        # 投影特征
        img_proj = self.image_proj(image_feat)  # [B, D/4]
        livox_proj = self.livox_proj(livox_feat)
        lidar360_proj = self.lidar360_proj(lidar360_feat)
        radar_proj = self.radar_proj(radar_feat)
        
        # 处理缺失：使用缺失嵌入替代（如果启用）
        if self.use_missing_embeddings and self.use_missing_masks:
            img_proj = img_proj * image_mask.unsqueeze(-1) + self.image_missing_emb.unsqueeze(0) * (1 - image_mask).unsqueeze(-1)
            livox_proj = livox_proj * livox_mask.unsqueeze(-1) + self.livox_missing_emb.unsqueeze(0) * (1 - livox_mask).unsqueeze(-1)
            lidar360_proj = lidar360_proj * lidar360_mask.unsqueeze(-1) + self.lidar360_missing_emb.unsqueeze(0) * (1 - lidar360_mask).unsqueeze(-1)
            radar_proj = radar_proj * radar_mask.unsqueeze(-1) + self.radar_missing_emb.unsqueeze(0) * (1 - radar_mask).unsqueeze(-1)
        elif self.use_missing_masks:
            # 只使用掩码，不使用缺失嵌入（用零向量替代）
            img_proj = img_proj * image_mask.unsqueeze(-1)
            livox_proj = livox_proj * livox_mask.unsqueeze(-1)
            lidar360_proj = lidar360_proj * lidar360_mask.unsqueeze(-1)
            radar_proj = radar_proj * radar_mask.unsqueeze(-1)
        else:
            # 不使用掩码和缺失嵌入（直接使用投影后的特征）
            pass
        
        # 计算综合权重：置信度权重 * 点云数量权重（如果启用）
        if self.use_weighting:
            # 置信度权重
            if self.use_confidence_weighting and self.confidence_proj is not None:
                img_conf_weight = self.confidence_proj(image_conf.unsqueeze(-1)).squeeze(-1) * image_mask
                livox_conf_weight = self.confidence_proj(livox_conf.unsqueeze(-1)).squeeze(-1) * livox_mask
                lidar360_conf_weight = self.confidence_proj(lidar360_conf.unsqueeze(-1)).squeeze(-1) * lidar360_mask
                radar_conf_weight = self.confidence_proj(radar_conf.unsqueeze(-1)).squeeze(-1) * radar_mask
            else:
                img_conf_weight = image_mask
                livox_conf_weight = livox_mask
                lidar360_conf_weight = lidar360_mask
                radar_conf_weight = radar_mask
            
            # 点云数量权重（归一化点云数量，然后映射到权重）
            if self.use_density_weighting and self.point_count_proj is not None:
                # 归一化点云数量到[0, 1]范围（使用log1p避免数值问题）
                max_point_count = torch.max(torch.stack([
                    image_point_count.max() if image_point_count.max() > 0 else torch.tensor(1.0, device=device),
                    livox_point_count.max() if livox_point_count.max() > 0 else torch.tensor(1.0, device=device),
                    lidar360_point_count.max() if lidar360_point_count.max() > 0 else torch.tensor(1.0, device=device),
                    radar_point_count.max() if radar_point_count.max() > 0 else torch.tensor(1.0, device=device),
                ]))
                
                img_point_weight = self.point_count_proj(torch.log1p(image_point_count / (max_point_count + 1e-8)).unsqueeze(-1)).squeeze(-1) * image_mask
                livox_point_weight = self.point_count_proj(torch.log1p(livox_point_count / (max_point_count + 1e-8)).unsqueeze(-1)).squeeze(-1) * livox_mask
                lidar360_point_weight = self.point_count_proj(torch.log1p(lidar360_point_count / (max_point_count + 1e-8)).unsqueeze(-1)).squeeze(-1) * lidar360_mask
                radar_point_weight = self.point_count_proj(torch.log1p(radar_point_count / (max_point_count + 1e-8)).unsqueeze(-1)).squeeze(-1) * radar_mask
            else:
                img_point_weight = image_mask
                livox_point_weight = livox_mask
                lidar360_point_weight = lidar360_mask
                radar_point_weight = radar_mask
            
            # 综合权重 = 置信度权重 * 点云数量权重（平衡两种因素）
            img_weight = img_conf_weight * img_point_weight
            livox_weight = livox_conf_weight * livox_point_weight
            lidar360_weight = lidar360_conf_weight * lidar360_point_weight
            radar_weight = radar_conf_weight * radar_point_weight
            
            # 归一化权重（使得总和为1）
            weight_sum = img_weight + livox_weight + lidar360_weight + radar_weight + 1e-8
            img_weight = img_weight / weight_sum
            livox_weight = livox_weight / weight_sum
            lidar360_weight = lidar360_weight / weight_sum
            radar_weight = radar_weight / weight_sum
        else:
            # 不使用加权（等权重）
            img_weight = torch.ones(B, device=device) / 4.0
            livox_weight = torch.ones(B, device=device) / 4.0
            lidar360_weight = torch.ones(B, device=device) / 4.0
            radar_weight = torch.ones(B, device=device) / 4.0
        
        # 加权特征
        img_proj = img_proj * img_weight.unsqueeze(-1)
        livox_proj = livox_proj * livox_weight.unsqueeze(-1)
        lidar360_proj = lidar360_proj * lidar360_weight.unsqueeze(-1)
        radar_proj = radar_proj * radar_weight.unsqueeze(-1)
        
        # 拼接四个模态
        fused = torch.cat([img_proj, livox_proj, lidar360_proj, radar_proj], dim=-1)  # [B, D]
        
        # 分类
        logits = self.fusion(fused)
        return logits


class CompactTransformer(nn.Module):
    """
    压缩版Transformer融合模型（四模态：图像、Livox、Lidar 360、Radar）
    
    参数量：~800,000
    适用场景：数据量 3,000-10,000
    
    支持显式缺失建模、置信度加权和点云数量加权
    """
    def __init__(
        self,
        image_dim: int = 768,
        livox_dim: int = 512,
        lidar360_dim: int = 512,
        radar_dim: int = 512,
        num_classes: int = 4,
        hidden_dim: int = 128,
        num_heads: int = 2,
        dropout: float = 0.3,
        use_missing_embeddings: bool = True,
        use_missing_masks: bool = True,
        use_weighting: bool = True,
        use_confidence_weighting: bool = True,
        use_density_weighting: bool = True,
    ):
        super().__init__()
        
        # 保存配置
        self.use_missing_embeddings = use_missing_embeddings
        self.use_missing_masks = use_missing_masks
        self.use_weighting = use_weighting
        self.use_confidence_weighting = use_confidence_weighting
        self.use_density_weighting = use_density_weighting
        
        # 特征投影
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.livox_proj = nn.Linear(livox_dim, hidden_dim)
        self.lidar360_proj = nn.Linear(lidar360_dim, hidden_dim)
        self.radar_proj = nn.Linear(radar_dim, hidden_dim)
        
        # 缺失嵌入（learnable embedding for missing modality）
        if use_missing_embeddings:
            self.image_missing_emb = nn.Parameter(torch.randn(hidden_dim))
            self.livox_missing_emb = nn.Parameter(torch.randn(hidden_dim))
            self.lidar360_missing_emb = nn.Parameter(torch.randn(hidden_dim))
            self.radar_missing_emb = nn.Parameter(torch.randn(hidden_dim))
        else:
            self.register_buffer('image_missing_emb', torch.zeros(hidden_dim))
            self.register_buffer('livox_missing_emb', torch.zeros(hidden_dim))
            self.register_buffer('lidar360_missing_emb', torch.zeros(hidden_dim))
            self.register_buffer('radar_missing_emb', torch.zeros(hidden_dim))
        
        # 位置编码（用于区分四个模态）
        self.modality_pos_emb = nn.Parameter(torch.randn(4, hidden_dim))
        
        # 单层自注意力（四模态交互）
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 置信度加权池化（如果启用）
        if use_confidence_weighting and use_weighting:
            self.confidence_proj = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        else:
            self.confidence_proj = None
        
        # 点云数量投影（如果启用）
        if use_density_weighting and use_weighting:
            self.point_count_proj = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
        else:
            self.point_count_proj = None
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 初始化缺失嵌入为小值（仅在启用且是Parameter时）
        if self.use_missing_embeddings and hasattr(self, 'image_missing_emb'):
            # 只初始化 Parameter 类型（Buffer 已经初始化为零，不需要重新初始化）
            if isinstance(self.image_missing_emb, nn.Parameter):
                nn.init.normal_(self.image_missing_emb, 0, 0.01)
                nn.init.normal_(self.livox_missing_emb, 0, 0.01)
                nn.init.normal_(self.lidar360_missing_emb, 0, 0.01)
                nn.init.normal_(self.radar_missing_emb, 0, 0.01)
        # 初始化位置编码
        if hasattr(self, 'modality_pos_emb') and isinstance(self.modality_pos_emb, nn.Parameter):
            nn.init.normal_(self.modality_pos_emb, 0, 0.01)
    
    def forward(
        self,
        image_feat: torch.Tensor,
        livox_feat: torch.Tensor,
        lidar360_feat: torch.Tensor,
        radar_feat: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        livox_mask: Optional[torch.Tensor] = None,
        lidar360_mask: Optional[torch.Tensor] = None,
        radar_mask: Optional[torch.Tensor] = None,
        image_conf: Optional[torch.Tensor] = None,
        livox_conf: Optional[torch.Tensor] = None,
        lidar360_conf: Optional[torch.Tensor] = None,
        radar_conf: Optional[torch.Tensor] = None,
        image_point_count: Optional[torch.Tensor] = None,
        livox_point_count: Optional[torch.Tensor] = None,
        lidar360_point_count: Optional[torch.Tensor] = None,
        radar_point_count: Optional[torch.Tensor] = None,
    ):
        B = image_feat.shape[0]
        device = image_feat.device
        
        # 检测缺失
        if image_mask is None:
            if self.use_missing_masks:
                image_mask = (image_feat.norm(dim=-1) > 1e-6).float()
            else:
                image_mask = torch.ones(B, device=device)
        else:
            image_mask = image_mask.to(device)
        if livox_mask is None:
            if self.use_missing_masks:
                livox_mask = (livox_feat.norm(dim=-1) > 1e-6).float()
            else:
                livox_mask = torch.ones(B, device=device)
        else:
            livox_mask = livox_mask.to(device)
        if lidar360_mask is None:
            if self.use_missing_masks:
                lidar360_mask = (lidar360_feat.norm(dim=-1) > 1e-6).float()
            else:
                lidar360_mask = torch.ones(B, device=device)
        else:
            lidar360_mask = lidar360_mask.to(device)
        if radar_mask is None:
            if self.use_missing_masks:
                radar_mask = (radar_feat.norm(dim=-1) > 1e-6).float()
            else:
                radar_mask = torch.ones(B, device=device)
        else:
            radar_mask = radar_mask.to(device)
        
        # 默认置信度，否则确保在正确设备上
        if image_conf is None:
            image_conf = torch.ones(B, device=device)
        else:
            image_conf = image_conf.to(device)
        if livox_conf is None:
            livox_conf = torch.ones(B, device=device)
        else:
            livox_conf = livox_conf.to(device)
        if lidar360_conf is None:
            lidar360_conf = torch.ones(B, device=device)
        else:
            lidar360_conf = lidar360_conf.to(device)
        if radar_conf is None:
            radar_conf = torch.ones(B, device=device)
        else:
            radar_conf = radar_conf.to(device)
        
        # 默认点云数量，否则确保在正确设备上
        if image_point_count is None:
            image_point_count = torch.zeros(B, device=device)
        else:
            image_point_count = image_point_count.to(device)
        if livox_point_count is None:
            livox_point_count = torch.zeros(B, device=device)
        else:
            livox_point_count = livox_point_count.to(device)
        if lidar360_point_count is None:
            lidar360_point_count = torch.zeros(B, device=device)
        else:
            lidar360_point_count = lidar360_point_count.to(device)
        if radar_point_count is None:
            radar_point_count = torch.zeros(B, device=device)
        else:
            radar_point_count = radar_point_count.to(device)
        
        # 投影特征
        img_proj = self.image_proj(image_feat)  # [B, D]
        livox_proj = self.livox_proj(livox_feat)
        lidar360_proj = self.lidar360_proj(lidar360_feat)
        radar_proj = self.radar_proj(radar_feat)
        
        # 处理缺失
        if self.use_missing_embeddings and self.use_missing_masks:
            img_proj = img_proj * image_mask.unsqueeze(-1) + self.image_missing_emb.unsqueeze(0) * (1 - image_mask).unsqueeze(-1)
            livox_proj = livox_proj * livox_mask.unsqueeze(-1) + self.livox_missing_emb.unsqueeze(0) * (1 - livox_mask).unsqueeze(-1)
            lidar360_proj = lidar360_proj * lidar360_mask.unsqueeze(-1) + self.lidar360_missing_emb.unsqueeze(0) * (1 - lidar360_mask).unsqueeze(-1)
            radar_proj = radar_proj * radar_mask.unsqueeze(-1) + self.radar_missing_emb.unsqueeze(0) * (1 - radar_mask).unsqueeze(-1)
        elif self.use_missing_masks:
            img_proj = img_proj * image_mask.unsqueeze(-1)
            livox_proj = livox_proj * livox_mask.unsqueeze(-1)
            lidar360_proj = lidar360_proj * lidar360_mask.unsqueeze(-1)
            radar_proj = radar_proj * radar_mask.unsqueeze(-1)
        
        # 添加位置编码
        img_proj = img_proj + self.modality_pos_emb[0:1]  # [B, D]
        livox_proj = livox_proj + self.modality_pos_emb[1:2]
        lidar360_proj = lidar360_proj + self.modality_pos_emb[2:3]
        radar_proj = radar_proj + self.modality_pos_emb[3:4]
        
        # 堆叠为序列 [B, 4, D]
        x = torch.stack([img_proj, livox_proj, lidar360_proj, radar_proj], dim=1)
        
        # 自注意力（四模态交互）
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)  # [B, 4, D]
        x = self.norm2(x + self.dropout(ffn_out))
        
        # 计算综合权重：置信度权重 * 点云数量权重（如果启用）
        if self.use_weighting:
            # 置信度权重
            if self.use_confidence_weighting and self.confidence_proj is not None:
                img_conf_weight = self.confidence_proj(image_conf.unsqueeze(-1).unsqueeze(-1)).squeeze(-1) * image_mask.unsqueeze(-1)
                livox_conf_weight = self.confidence_proj(livox_conf.unsqueeze(-1).unsqueeze(-1)).squeeze(-1) * livox_mask.unsqueeze(-1)
                lidar360_conf_weight = self.confidence_proj(lidar360_conf.unsqueeze(-1).unsqueeze(-1)).squeeze(-1) * lidar360_mask.unsqueeze(-1)
                radar_conf_weight = self.confidence_proj(radar_conf.unsqueeze(-1).unsqueeze(-1)).squeeze(-1) * radar_mask.unsqueeze(-1)
            else:
                img_conf_weight = image_mask.unsqueeze(-1)
                livox_conf_weight = livox_mask.unsqueeze(-1)
                lidar360_conf_weight = lidar360_mask.unsqueeze(-1)
                radar_conf_weight = radar_mask.unsqueeze(-1)
            
            # 点云数量权重
            if self.use_density_weighting and self.point_count_proj is not None:
                max_point_count = torch.max(torch.stack([
                    image_point_count.max() if image_point_count.max() > 0 else torch.tensor(1.0, device=device),
                    livox_point_count.max() if livox_point_count.max() > 0 else torch.tensor(1.0, device=device),
                    lidar360_point_count.max() if lidar360_point_count.max() > 0 else torch.tensor(1.0, device=device),
                    radar_point_count.max() if radar_point_count.max() > 0 else torch.tensor(1.0, device=device),
                ]))
                
                img_point_weight = self.point_count_proj(torch.log1p(image_point_count / (max_point_count + 1e-8)).unsqueeze(-1).unsqueeze(-1)).squeeze(-1) * image_mask.unsqueeze(-1)
                livox_point_weight = self.point_count_proj(torch.log1p(livox_point_count / (max_point_count + 1e-8)).unsqueeze(-1).unsqueeze(-1)).squeeze(-1) * livox_mask.unsqueeze(-1)
                lidar360_point_weight = self.point_count_proj(torch.log1p(lidar360_point_count / (max_point_count + 1e-8)).unsqueeze(-1).unsqueeze(-1)).squeeze(-1) * lidar360_mask.unsqueeze(-1)
                radar_point_weight = self.point_count_proj(torch.log1p(radar_point_count / (max_point_count + 1e-8)).unsqueeze(-1).unsqueeze(-1)).squeeze(-1) * radar_mask.unsqueeze(-1)
            else:
                img_point_weight = image_mask.unsqueeze(-1)
                livox_point_weight = livox_mask.unsqueeze(-1)
                lidar360_point_weight = lidar360_mask.unsqueeze(-1)
                radar_point_weight = radar_mask.unsqueeze(-1)
            
            img_weight = img_conf_weight * img_point_weight
            livox_weight = livox_conf_weight * livox_point_weight
            lidar360_weight = lidar360_conf_weight * lidar360_point_weight
            radar_weight = radar_conf_weight * radar_point_weight
            
            weights = torch.cat([img_weight, livox_weight, lidar360_weight, radar_weight], dim=1)  # [B, 4]
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        else:
            # 不使用加权（等权重）
            weights = torch.ones(B, 4, device=device) / 4.0
        
        # 加权平均池化
        x = (x * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        
        # 分类
        logits = self.classifier(x)
        return logits


class EfficientFusion(nn.Module):
    """
    高效融合模型（四模态：图像、Livox、Lidar 360、Radar）
    
    参数量：~3,000,000
    适用场景：数据量 10,000+
    
    支持显式缺失建模、置信度加权和点云数量加权，使用交叉注意力进行多模态交互
    """
    def __init__(
        self,
        image_dim: int = 768,
        livox_dim: int = 512,
        lidar360_dim: int = 512,
        radar_dim: int = 512,
        num_classes: int = 4,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_missing_embeddings: bool = True,
        use_missing_masks: bool = True,
        use_weighting: bool = True,
        use_confidence_weighting: bool = True,
        use_density_weighting: bool = True,
    ):
        super().__init__()
        
        # 保存配置
        self.use_missing_embeddings = use_missing_embeddings
        self.use_missing_masks = use_missing_masks
        self.use_weighting = use_weighting
        self.use_confidence_weighting = use_confidence_weighting
        self.use_density_weighting = use_density_weighting
        
        # 特征投影（带LayerNorm）
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.livox_proj = nn.Sequential(
            nn.Linear(livox_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.lidar360_proj = nn.Sequential(
            nn.Linear(lidar360_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.radar_proj = nn.Sequential(
            nn.Linear(radar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 缺失嵌入
        if use_missing_embeddings:
            self.image_missing_emb = nn.Parameter(torch.randn(hidden_dim))
            self.livox_missing_emb = nn.Parameter(torch.randn(hidden_dim))
            self.lidar360_missing_emb = nn.Parameter(torch.randn(hidden_dim))
            self.radar_missing_emb = nn.Parameter(torch.randn(hidden_dim))
        else:
            self.register_buffer('image_missing_emb', torch.zeros(hidden_dim))
            self.register_buffer('livox_missing_emb', torch.zeros(hidden_dim))
            self.register_buffer('lidar360_missing_emb', torch.zeros(hidden_dim))
            self.register_buffer('radar_missing_emb', torch.zeros(hidden_dim))
        
        # 多模态交叉注意力（图像作为query，点云作为key/value）
        self.image_to_pointcloud_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 点云模态间交叉注意力（Livox、Lidar 360和Radar交互）
        self.pointcloud_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # 置信度加权融合（如果启用）
        if use_confidence_weighting and use_weighting:
            self.confidence_proj = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            self.confidence_proj = None
        
        # 点云数量投影（如果启用）
        if use_density_weighting and use_weighting:
            self.point_count_proj = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            self.point_count_proj = None
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # 初始化缺失嵌入为小值（仅在启用且是Parameter时）
        if self.use_missing_embeddings and hasattr(self, 'image_missing_emb'):
            # 只初始化 Parameter 类型（Buffer 已经初始化为零，不需要重新初始化）
            if isinstance(self.image_missing_emb, nn.Parameter):
                nn.init.normal_(self.image_missing_emb, 0, 0.01)
                nn.init.normal_(self.livox_missing_emb, 0, 0.01)
                nn.init.normal_(self.lidar360_missing_emb, 0, 0.01)
                nn.init.normal_(self.radar_missing_emb, 0, 0.01)
    
    def forward(
        self,
        image_feat: torch.Tensor,
        livox_feat: torch.Tensor,
        lidar360_feat: torch.Tensor,
        radar_feat: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        livox_mask: Optional[torch.Tensor] = None,
        lidar360_mask: Optional[torch.Tensor] = None,
        radar_mask: Optional[torch.Tensor] = None,
        image_conf: Optional[torch.Tensor] = None,
        livox_conf: Optional[torch.Tensor] = None,
        lidar360_conf: Optional[torch.Tensor] = None,
        radar_conf: Optional[torch.Tensor] = None,
        image_point_count: Optional[torch.Tensor] = None,
        livox_point_count: Optional[torch.Tensor] = None,
        lidar360_point_count: Optional[torch.Tensor] = None,
        radar_point_count: Optional[torch.Tensor] = None,
    ):
        B = image_feat.shape[0]
        device = image_feat.device
        
        # 检测缺失
        if image_mask is None:
            if self.use_missing_masks:
                image_mask = (image_feat.norm(dim=-1) > 1e-6).float()
            else:
                image_mask = torch.ones(B, device=device)
        else:
            image_mask = image_mask.to(device)
        if livox_mask is None:
            if self.use_missing_masks:
                livox_mask = (livox_feat.norm(dim=-1) > 1e-6).float()
            else:
                livox_mask = torch.ones(B, device=device)
        else:
            livox_mask = livox_mask.to(device)
        if lidar360_mask is None:
            if self.use_missing_masks:
                lidar360_mask = (lidar360_feat.norm(dim=-1) > 1e-6).float()
            else:
                lidar360_mask = torch.ones(B, device=device)
        else:
            lidar360_mask = lidar360_mask.to(device)
        if radar_mask is None:
            if self.use_missing_masks:
                radar_mask = (radar_feat.norm(dim=-1) > 1e-6).float()
            else:
                radar_mask = torch.ones(B, device=device)
        else:
            radar_mask = radar_mask.to(device)
        
        # 默认置信度，否则确保在正确设备上
        if image_conf is None:
            image_conf = torch.ones(B, device=device)
        else:
            image_conf = image_conf.to(device)
        if livox_conf is None:
            livox_conf = torch.ones(B, device=device)
        else:
            livox_conf = livox_conf.to(device)
        if lidar360_conf is None:
            lidar360_conf = torch.ones(B, device=device)
        else:
            lidar360_conf = lidar360_conf.to(device)
        if radar_conf is None:
            radar_conf = torch.ones(B, device=device)
        else:
            radar_conf = radar_conf.to(device)
        
        # 默认点云数量，否则确保在正确设备上
        if image_point_count is None:
            image_point_count = torch.zeros(B, device=device)
        else:
            image_point_count = image_point_count.to(device)
        if livox_point_count is None:
            livox_point_count = torch.zeros(B, device=device)
        else:
            livox_point_count = livox_point_count.to(device)
        if lidar360_point_count is None:
            lidar360_point_count = torch.zeros(B, device=device)
        else:
            lidar360_point_count = lidar360_point_count.to(device)
        if radar_point_count is None:
            radar_point_count = torch.zeros(B, device=device)
        else:
            radar_point_count = radar_point_count.to(device)
        
        # 投影特征（EfficientFusion使用Sequential，需要调用）
        img_proj = self.image_proj(image_feat)  # [B, D]
        livox_proj = self.livox_proj(livox_feat)
        lidar360_proj = self.lidar360_proj(lidar360_feat)
        radar_proj = self.radar_proj(radar_feat)
        
        # 处理缺失
        if self.use_missing_embeddings and self.use_missing_masks:
            img_proj = img_proj * image_mask.unsqueeze(-1) + self.image_missing_emb.unsqueeze(0) * (1 - image_mask).unsqueeze(-1)
            livox_proj = livox_proj * livox_mask.unsqueeze(-1) + self.livox_missing_emb.unsqueeze(0) * (1 - livox_mask).unsqueeze(-1)
            lidar360_proj = lidar360_proj * lidar360_mask.unsqueeze(-1) + self.lidar360_missing_emb.unsqueeze(0) * (1 - lidar360_mask).unsqueeze(-1)
            radar_proj = radar_proj * radar_mask.unsqueeze(-1) + self.radar_missing_emb.unsqueeze(0) * (1 - radar_mask).unsqueeze(-1)
        elif self.use_missing_masks:
            img_proj = img_proj * image_mask.unsqueeze(-1)
            livox_proj = livox_proj * livox_mask.unsqueeze(-1)
            lidar360_proj = lidar360_proj * lidar360_mask.unsqueeze(-1)
            radar_proj = radar_proj * radar_mask.unsqueeze(-1)
        
        # 图像到点云的交叉注意力（图像作为query，点云作为key/value）
        pointcloud_kv = torch.stack([livox_proj, lidar360_proj, radar_proj], dim=1)  # [B, 3, D]
        img_query = img_proj.unsqueeze(1)  # [B, 1, D]
        
        attn_out1, _ = self.image_to_pointcloud_attn(img_query, pointcloud_kv, pointcloud_kv)
        img_enhanced = self.norm1(img_query + self.dropout(attn_out1))  # [B, 1, D]
        img_enhanced = img_enhanced.squeeze(1)  # [B, D]
        
        # 点云模态间交叉注意力（Livox、Lidar 360和Radar交互）
        pointcloud_features = torch.stack([livox_proj, lidar360_proj, radar_proj], dim=1)  # [B, 3, D]
        attn_out2, _ = self.pointcloud_cross_attn(pointcloud_features, pointcloud_features, pointcloud_features)
        pointcloud_enhanced = self.norm2(pointcloud_features + self.dropout(attn_out2))  # [B, 3, D]
        
        livox_enhanced = pointcloud_enhanced[:, 0, :]  # [B, D]
        lidar360_enhanced = pointcloud_enhanced[:, 1, :]  # [B, D]
        radar_enhanced = pointcloud_enhanced[:, 2, :]  # [B, D]
        
        # Feed-forward
        img_enhanced = self.norm3(img_enhanced + self.ffn(img_enhanced))
        livox_enhanced = self.norm3(livox_enhanced + self.ffn(livox_enhanced))
        lidar360_enhanced = self.norm3(lidar360_enhanced + self.ffn(lidar360_enhanced))
        radar_enhanced = self.norm3(radar_enhanced + self.ffn(radar_enhanced))
        
        # 计算综合权重：置信度权重 * 点云数量权重（如果启用）
        if self.use_weighting:
            # 置信度权重
            if self.use_confidence_weighting and self.confidence_proj is not None:
                img_conf_weight = self.confidence_proj(image_conf.unsqueeze(-1)).squeeze(-1) * image_mask
                livox_conf_weight = self.confidence_proj(livox_conf.unsqueeze(-1)).squeeze(-1) * livox_mask
                lidar360_conf_weight = self.confidence_proj(lidar360_conf.unsqueeze(-1)).squeeze(-1) * lidar360_mask
                radar_conf_weight = self.confidence_proj(radar_conf.unsqueeze(-1)).squeeze(-1) * radar_mask
            else:
                img_conf_weight = image_mask
                livox_conf_weight = livox_mask
                lidar360_conf_weight = lidar360_mask
                radar_conf_weight = radar_mask
            
            # 点云数量权重
            if self.use_density_weighting and self.point_count_proj is not None:
                max_point_count = torch.max(torch.stack([
                    image_point_count.max() if image_point_count.max() > 0 else torch.tensor(1.0, device=device),
                    livox_point_count.max() if livox_point_count.max() > 0 else torch.tensor(1.0, device=device),
                    lidar360_point_count.max() if lidar360_point_count.max() > 0 else torch.tensor(1.0, device=device),
                    radar_point_count.max() if radar_point_count.max() > 0 else torch.tensor(1.0, device=device),
                ]))
                
                img_point_weight = self.point_count_proj(torch.log1p(image_point_count / (max_point_count + 1e-8)).unsqueeze(-1)).squeeze(-1) * image_mask
                livox_point_weight = self.point_count_proj(torch.log1p(livox_point_count / (max_point_count + 1e-8)).unsqueeze(-1)).squeeze(-1) * livox_mask
                lidar360_point_weight = self.point_count_proj(torch.log1p(lidar360_point_count / (max_point_count + 1e-8)).unsqueeze(-1)).squeeze(-1) * lidar360_mask
                radar_point_weight = self.point_count_proj(torch.log1p(radar_point_count / (max_point_count + 1e-8)).unsqueeze(-1)).squeeze(-1) * radar_mask
            else:
                img_point_weight = image_mask
                livox_point_weight = livox_mask
                lidar360_point_weight = lidar360_mask
                radar_point_weight = radar_mask
            
            img_weight = img_conf_weight * img_point_weight
            livox_weight = livox_conf_weight * livox_point_weight
            lidar360_weight = lidar360_conf_weight * lidar360_point_weight
            radar_weight = radar_conf_weight * radar_point_weight
            
            weight_sum = img_weight + livox_weight + lidar360_weight + radar_weight + 1e-8
            img_weight = img_weight / weight_sum
            livox_weight = livox_weight / weight_sum
            lidar360_weight = lidar360_weight / weight_sum
            radar_weight = radar_weight / weight_sum
        else:
            # 不使用加权（等权重）
            img_weight = torch.ones(B, device=device) / 4.0
            livox_weight = torch.ones(B, device=device) / 4.0
            lidar360_weight = torch.ones(B, device=device) / 4.0
            radar_weight = torch.ones(B, device=device) / 4.0
        
        # 加权融合
        fused = (img_enhanced * img_weight.unsqueeze(-1) +
                livox_enhanced * livox_weight.unsqueeze(-1) +
                lidar360_enhanced * lidar360_weight.unsqueeze(-1) +
                radar_enhanced * radar_weight.unsqueeze(-1))  # [B, D]
        
        # 分类
        logits = self.classifier(fused)
        return logits


def create_lightweight_classifier(
    model_type: Literal["tiny", "compact", "efficient"] = "compact",
    image_dim: int = 768,
    livox_dim: int = 512,
    lidar360_dim: int = 512,
    radar_dim: int = 512,
    num_classes: int = 4,
    **kwargs
) -> nn.Module:
    """
    创建轻量级分类器（四模态：图像、Livox、Lidar 360、Radar）
    
    Args:
        model_type: 模型类型
            - "tiny": 超轻量级MLP (~100K参数)
            - "compact": 压缩版Transformer (~800K参数)
            - "efficient": 平衡版本 (~3M参数)
        image_dim: 图像特征维度（ConvNeXt，默认768）
        livox_dim: Livox点云特征维度（PointNeXt，默认512）
        lidar360_dim: Lidar 360点云特征维度（PointNeXt，默认512）
        radar_dim: Radar点云特征维度（PointNeXt，默认512）
        num_classes: 类别数量
        **kwargs: 其他模型参数
    
    Returns:
        轻量级分类器模型
    """
    if model_type == "tiny":
        return TinyFusion(
            image_dim=image_dim,
            livox_dim=livox_dim,
            lidar360_dim=lidar360_dim,
            radar_dim=radar_dim,
            num_classes=num_classes,
            hidden_dim=kwargs.get('hidden_dim', 128),
            dropout=kwargs.get('dropout', 0.5)
        )
    elif model_type == "compact":
        return CompactTransformer(
            image_dim=image_dim,
            livox_dim=livox_dim,
            lidar360_dim=lidar360_dim,
            radar_dim=radar_dim,
            num_classes=num_classes,
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_heads=kwargs.get('num_heads', 2),
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_type == "efficient":
        return EfficientFusion(
            image_dim=image_dim,
            livox_dim=livox_dim,
            lidar360_dim=lidar360_dim,
            radar_dim=radar_dim,
            num_classes=num_classes,
            hidden_dim=kwargs.get('hidden_dim', 256),
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试不同模型的参数量
    print("=" * 80)
    print("轻量级四模态融合模型参数量对比")
    print("=" * 80)
    
    models = {
        "Tiny MLP": create_lightweight_classifier("tiny"),
        "Compact Transformer": create_lightweight_classifier("compact"),
        "Efficient Fusion": create_lightweight_classifier("efficient"),
    }
    
    for name, model in models.items():
        params = count_parameters(model)
        print(f"\n{name}:")
        print(f"  参数量: {params:,}")
        print(f"  适用数据量: ", end="")
        if params < 150000:
            print("< 3,000 样本")
        elif params < 1000000:
            print("3,000-10,000 样本")
        else:
            print("> 10,000 样本")
        
        # 测试前向传播（四模态）
        image_feat = torch.randn(4, 768)
        livox_feat = torch.randn(4, 512)
        lidar360_feat = torch.randn(4, 512)
        radar_feat = torch.randn(4, 512)
        
        # 测试完整模态
        output = model(image_feat, livox_feat, lidar360_feat, radar_feat)
        print(f"  输出形状（完整模态）: {output.shape}")
        
        # 测试缺失模态
        image_mask = torch.tensor([1., 1., 0., 0.])  # 后两个样本缺失图像
        livox_mask = torch.tensor([1., 0., 1., 0.])  # 第2、4个样本缺失Livox
        lidar360_mask = torch.tensor([1., 1., 1., 0.])  # 第4个样本缺失Lidar 360
        radar_mask = torch.tensor([1., 1., 1., 1.])  # 所有样本都有Radar
        
        output_missing = model(
            image_feat, livox_feat, lidar360_feat, radar_feat,
            image_mask=image_mask,
            livox_mask=livox_mask,
            lidar360_mask=lidar360_mask,
            radar_mask=radar_mask
        )
        print(f"  输出形状（部分缺失）: {output_missing.shape}")
    
    print("\n" + "=" * 80)
    print("推荐：")
    print("  当前数据集（2,183样本）→ 使用 Compact Transformer")
    print("  重组数据集（10,000+样本）→ 使用 Efficient Fusion")
    print("=" * 80)

