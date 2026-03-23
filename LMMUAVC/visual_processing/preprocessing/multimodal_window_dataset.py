"""
多模态时间窗口数据集：在0.4秒窗口内聚合点云和图像特征。

该数据集类支持：
1. 时间窗口对齐（0.4秒窗口）
2. 加载预提取的PointNeXt和ConvNeXt特征
3. 特征聚合（平均池化、最大池化、注意力聚合等）
4. 类别标签加载
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Literal
from collections import OrderedDict


class MultimodalWindowDataset(Dataset):
    """
    多模态时间窗口数据集。
    
    在指定的时间窗口（默认0.4秒）内聚合点云和图像特征。
    """

    def __init__(
        self,
        timeline_dir: Path,
        base_dir: Path,
        image_features_path: Optional[Path] = None,  # ConvNeXt特征（图像）
        livox_features_path: Optional[Path] = None,  # Livox PointNeXt特征
        lidar360_features_path: Optional[Path] = None,  # Lidar 360 PointNeXt特征
        radar_features_path: Optional[Path] = None,  # Radar PointNeXt特征
        # 为了向后兼容，保留旧的参数名
        pointnext_features_path: Optional[Path] = None,  # 旧参数名，等同于livox_features_path
        convnext_features_path: Optional[Path] = None,  # 旧参数名，等同于image_features_path
        train: bool = True,
        window_size: float = 0.4,
        aggregation_mode: Literal["mean", "max", "attention", "concat"] = "mean",
        normalize_features: bool = True,
        use_precomputed_features: bool = True,
        split_config_path: Optional[Path] = None,
        target_split: Optional[str] = None,
        require_all_modalities: bool = False,  # 是否要求所有模态都存在（默认False，允许缺失）
    ):
        """
        Args:
            timeline_dir: 对齐后的时间线CSV文件目录（包含train/val子目录）
            base_dir: 数据集根目录（包含train/val子目录）
            image_features_path: 预提取的图像特征文件路径（ConvNeXt，*.pt）
            livox_features_path: 预提取的Livox点云特征文件路径（PointNeXt，*.pt）
            lidar360_features_path: 预提取的Lidar 360点云特征文件路径（PointNeXt，*.pt）
            radar_features_path: 预提取的Radar点云特征文件路径（PointNeXt，*.pt）
            pointnext_features_path: （向后兼容）等同于livox_features_path
            convnext_features_path: （向后兼容）等同于image_features_path
            train: 是否为训练集
            window_size: 时间窗口大小（秒），默认0.4秒
            aggregation_mode: 特征聚合模式
                - "mean": 平均池化
                - "max": 最大池化
                - "attention": 注意力加权聚合
                - "concat": 拼接所有特征（不聚合）
            normalize_features: 是否对特征进行L2归一化
            use_precomputed_features: 是否使用预计算的特征（True）或实时提取（False）
            split_config_path: 数据集划分配置文件路径（用于软件层面重组）
            target_split: 目标数据集分割（train/val/test），用于软件层面重组
            require_all_modalities: 是否只保留同时有所有模态特征的样本（默认False，允许缺失）
        """
        self.split_config = None
        self.split_lookup = {}
        self.target_split = target_split
        self.require_all_modalities = require_all_modalities
        
        # 加载划分配置（如果提供）
        if split_config_path and Path(split_config_path).exists():
            import json
            with open(split_config_path, 'r', encoding='utf-8') as f:
                self.split_config = json.load(f)
            
            if 'split_lookup' in self.split_config and target_split in self.split_config['split_lookup']:
                self.split_lookup = self.split_config['split_lookup'][target_split]
                print(f"✅ 已加载数据集划分配置: {target_split} 集包含 {len(self.split_lookup)} 个样本")
                print(f"   类别分布: {self.split_config['statistics'][target_split]['class_distribution']}")
        
        # 确定实际使用的split（支持test split）
        if target_split and target_split in ['train', 'val', 'test']:
            actual_split = target_split
        else:
            actual_split = 'train' if train else 'val'
        
        self.base_dir = base_dir / actual_split
        self.timeline_dir = timeline_dir / actual_split
        self.window_size = window_size
        self.aggregation_mode = aggregation_mode
        self.normalize_features = normalize_features
        self.use_precomputed_features = use_precomputed_features

        # 加载对齐后的时间线
        self.timeline = self._load_timeline()
        
        # 如果使用划分配置，过滤时间线
        if self.split_config and self.split_lookup:
            self._filter_timeline_by_split()
        
        # 向后兼容：如果使用旧参数名，映射到新参数
        if pointnext_features_path and not livox_features_path:
            livox_features_path = pointnext_features_path
        if convnext_features_path and not image_features_path:
            image_features_path = convnext_features_path
        
        # 加载预计算的特征（四种模态）
        self.image_features = None  # ConvNeXt特征（图像）
        self.livox_features = None  # Livox点云特征
        self.lidar360_features = None  # Lidar 360点云特征
        self.radar_features = None  # Radar点云特征
        self.image_metadata = []
        self.livox_metadata = []
        self.lidar360_metadata = []
        self.radar_metadata = []
        
        # 为了向后兼容，保留旧的属性名
        self.convnext_features = None
        self.pointnext_features = None
        self.convnext_metadata = []
        self.pointnext_metadata = []
        
        if use_precomputed_features:
            if image_features_path:
                self._load_image_features(image_features_path)
            if livox_features_path:
                self._load_livox_features(livox_features_path)
            if lidar360_features_path:
                self._load_lidar360_features(lidar360_features_path)
            if radar_features_path:
                self._load_radar_features(radar_features_path)
        
        # 构建时间窗口索引
        # 如果使用预计算特征且特征文件已经划分好，直接使用特征文件的长度
        if use_precomputed_features and target_split:
            # 检查是否有特征文件（至少一个模态）
            has_features = any([
                self.image_features is not None,
                self.livox_features is not None,
                self.lidar360_features is not None,
                self.radar_features is not None
            ])
            
            if has_features:
                # 使用特征文件的长度（所有模态应该相同）
                feature_length = None
                metadata_list = []
                if self.image_features is not None:
                    feature_length = len(self.image_features)
                    metadata_list = self.image_metadata
                elif self.livox_features is not None:
                    feature_length = len(self.livox_features)
                    metadata_list = self.livox_metadata
                elif self.lidar360_features is not None:
                    feature_length = len(self.lidar360_features)
                    metadata_list = self.lidar360_metadata
                elif self.radar_features is not None:
                    feature_length = len(self.radar_features)
                    metadata_list = self.radar_metadata
                
                if feature_length is not None and feature_length > 0:
                    # 直接创建索引，每个样本对应一个窗口（使用特征文件索引，而不是timeline索引）
                    self.window_indices = []
                    for i in range(feature_length):
                        center_time = 0.0
                        if metadata_list and i < len(metadata_list) and isinstance(metadata_list[i], dict):
                            center_time = metadata_list[i].get('window_center', metadata_list[i].get('timestamp', 0.0))
                        self.window_indices.append({
                            'center_idx': i,  # 特征文件中的索引
                            'feature_idx': i,  # 特征文件中的索引（直接使用，不通过timeline）
                            'window_indices': [i],  # 单个样本窗口
                            'center_time': center_time,
                            'use_feature_idx_directly': True  # 标记：直接使用特征索引
                        })
                    print(f"✅ 使用预计算特征文件长度: {feature_length} 个样本（{target_split}集），直接使用特征索引")
                else:
                    self.window_indices = self._build_window_indices()
            else:
                self.window_indices = self._build_window_indices()
        else:
            self.window_indices = self._build_window_indices()
        
        # 如果需要，过滤掉没有完整特征的样本（要求所有模态都存在）
        if require_all_modalities and use_precomputed_features:
            self._filter_complete_features()
        
        # 预加载类别文件的时间戳索引（提高匹配效率，解决时间戳精度问题）
        self.class_file_cache = {}  # (seq, timestamp) -> file_path
        self._build_class_file_cache()

    def _filter_timeline_by_split(self):
        """根据划分配置过滤时间线（软件层面重组）。"""
        if not self.split_lookup:
            return
        
        print(f"正在根据划分配置过滤时间线...")
        filtered_indices = []
        
        for idx, row in self.timeline.iterrows():
            try:
                seq = int(row['seq'])
                timestamp = float(row.get('class', 0))  # class列是时间戳
                
                # 尝试所有可能的原始split来构建查找键
                for original_split in ['train', 'val', 'test', 'all']:
                    lookup_key = f"{original_split}/seq{seq:04d}/{timestamp:.7f}"
                    
                    if lookup_key in self.split_lookup:
                        # 找到匹配的样本
                        filtered_indices.append(idx)
                        break
            except Exception as e:
                continue
        
        if filtered_indices:
            self.timeline = self.timeline.iloc[filtered_indices].reset_index(drop=True)
            print(f"✅ 过滤后的时间线: {len(self.timeline)} 个样本")
        else:
            print(f"⚠️  警告: 根据划分配置未找到任何匹配的样本")

    def _load_timeline(self) -> pd.DataFrame:
        """加载对齐后的时间线CSV文件。"""
        result = None
        csv_files = sorted(
            self.timeline_dir.glob("*.csv"),
            key=lambda x: int(x.stem) if x.stem.isdigit() else 0
        )
        
        for file_path in csv_files:
            if result is None:
                result = pd.read_csv(file_path, delimiter='\t', dtype='str')
            else:
                result = pd.concat([result, pd.read_csv(file_path, delimiter='\t', dtype='str')])
        
        if result is None:
            raise FileNotFoundError(f"未找到时间线文件: {self.timeline_dir}")
        
        # 转换时间戳列为float
        time_columns = ['average', 'lidar_360', 'Image', 'livox_avia', 'ground_truth', 'radar_enhance_pcl', 'class']
        for col in time_columns:
            if col in result.columns:
                result[col] = result[col].astype(float)
        
        return result.reset_index(drop=True)

    def _load_image_features(self, feature_path: Path):
        """加载图像特征（ConvNeXt）。"""
        data = torch.load(feature_path, map_location='cpu')
        self.image_features = data['features']  # (N, D)
        self.image_metadata = data.get('metadata', [])
        # 向后兼容
        self.convnext_features = self.image_features
        self.convnext_metadata = self.image_metadata
        print(f"已加载图像特征（ConvNeXt）: {self.image_features.shape}")

    def _load_livox_features(self, feature_path: Path):
        """加载Livox点云特征（PointNeXt）。"""
        data = torch.load(feature_path, map_location='cpu')
        self.livox_features = data['features']  # (N, D)
        self.livox_metadata = data.get('metadata', [])
        # 向后兼容
        self.pointnext_features = self.livox_features
        self.pointnext_metadata = self.livox_metadata
        print(f"已加载Livox点云特征（PointNeXt）: {self.livox_features.shape}")

    def _load_lidar360_features(self, feature_path: Path):
        """加载Lidar 360点云特征（PointNeXt）。"""
        data = torch.load(feature_path, map_location='cpu')
        self.lidar360_features = data['features']  # (N, D)
        self.lidar360_metadata = data.get('metadata', [])
        print(f"已加载Lidar 360点云特征（PointNeXt）: {self.lidar360_features.shape}")

    def _load_radar_features(self, feature_path: Path):
        """加载Radar点云特征（PointNeXt）。"""
        data = torch.load(feature_path, map_location='cpu')
        self.radar_features = data['features']  # (N, D)
        self.radar_metadata = data.get('metadata', [])
        print(f"已加载Radar点云特征（PointNeXt）: {self.radar_features.shape}")

    def _build_window_indices(self) -> List[Dict]:
        """
        构建时间窗口索引。
        
        对于每个基准时间戳，找到窗口内的所有样本索引。
        """
        window_indices = []
        base_timestamps = self.timeline['average'].values
        
        for i, center_time in enumerate(base_timestamps):
            # 找到窗口内的所有时间戳
            window_start = center_time - self.window_size / 2
            window_end = center_time + self.window_size / 2
            
            # 找到窗口内的索引
            mask = (base_timestamps >= window_start) & (base_timestamps <= window_end)
            indices = np.where(mask)[0].tolist()
            
            window_indices.append({
                'center_idx': i,
                'window_indices': indices,
                'center_time': center_time,
            })
        
        return window_indices

    def _filter_complete_features(self):
        """过滤掉没有完整特征（所有模态都存在）的样本"""
        if not self.use_precomputed_features:
            return
        
        original_count = len(self.window_indices)
        filtered_indices = []
        
        print(f"\n过滤没有完整特征的样本（要求所有模态都存在）...")
        print(f"  原始样本数: {original_count}")
        
        for idx, window_info in enumerate(self.window_indices):
            center_idx = window_info['center_idx']
            window_idxs = window_info['window_indices']
            
            # 检查窗口内是否有所有模态的特征
            has_image = False
            has_livox = False
            has_lidar360 = False
            has_radar = False
            
            for win_idx in window_idxs:
                row = self.timeline.iloc[win_idx]
                
                # 检查图像特征
                if self.image_features is not None:
                    img_idx = self._match_feature_index(win_idx, row, "image")
                    if img_idx is not None and img_idx < len(self.image_features):
                        if self.image_features[img_idx].norm() > 1e-6:
                            has_image = True
                
                # 检查Livox特征
                if self.livox_features is not None:
                    livox_idx = self._match_feature_index(win_idx, row, "livox")
                    if livox_idx is not None and livox_idx < len(self.livox_features):
                        if self.livox_features[livox_idx].norm() > 1e-6:
                            has_livox = True
                
                # 检查Lidar 360特征
                if self.lidar360_features is not None:
                    lidar360_idx = self._match_feature_index(win_idx, row, "lidar360")
                    if lidar360_idx is not None and lidar360_idx < len(self.lidar360_features):
                        if self.lidar360_features[lidar360_idx].norm() > 1e-6:
                            has_lidar360 = True
                
                # 检查Radar特征
                if self.radar_features is not None:
                    radar_idx = self._match_feature_index(win_idx, row, "radar")
                    if radar_idx is not None and radar_idx < len(self.radar_features):
                        if self.radar_features[radar_idx].norm() > 1e-6:
                            has_radar = True
                
                # 检查是否所有模态都存在（如果某个模态的特征文件未加载，则跳过该模态的检查）
                required_modalities = []
                if self.image_features is not None:
                    required_modalities.append(has_image)
                if self.livox_features is not None:
                    required_modalities.append(has_livox)
                if self.lidar360_features is not None:
                    required_modalities.append(has_lidar360)
                if self.radar_features is not None:
                    required_modalities.append(has_radar)
                
                if required_modalities and all(required_modalities):
                    break
            
            # 只保留有所有模态特征的样本（如果某个模态的特征文件未加载，则跳过该模态的检查）
            required_modalities = []
            if self.image_features is not None:
                required_modalities.append(has_image)
            if self.livox_features is not None:
                required_modalities.append(has_livox)
            if self.lidar360_features is not None:
                required_modalities.append(has_lidar360)
            if self.radar_features is not None:
                required_modalities.append(has_radar)
            
            if required_modalities and all(required_modalities):
                filtered_indices.append(window_info)
        
        self.window_indices = filtered_indices
        filtered_count = len(self.window_indices)
        removed_count = original_count - filtered_count
        
        print(f"  过滤后样本数: {filtered_count}")
        print(f"  移除样本数: {removed_count} ({100*removed_count/original_count:.1f}%)")
        
        if filtered_count == 0:
            print(f"  ⚠️  警告：过滤后没有样本！请检查特征文件")

    def _match_feature_index(
        self,
        timeline_idx: int,
        row: pd.Series,
        feature_type: Literal["pointnext", "convnext", "image", "livox", "lidar360", "radar"]
    ) -> Optional[int]:
        """
        匹配特征索引。
        
        支持两种模式：
        1. 检测后的特征：根据时间戳和序列号匹配
        2. 原始特征：根据时间戳匹配
        
        优先使用时间戳匹配，如果失败则使用索引匹配。
        """
        # 映射特征类型
        if feature_type == "pointnext" or feature_type == "livox":
            metadata = self.livox_metadata if hasattr(self, 'livox_metadata') else self.pointnext_metadata
        elif feature_type == "convnext" or feature_type == "image":
            metadata = self.image_metadata if hasattr(self, 'image_metadata') else self.convnext_metadata
        elif feature_type == "lidar360":
            metadata = self.lidar360_metadata
        elif feature_type == "radar":
            metadata = self.radar_metadata if hasattr(self, 'radar_metadata') else []
        else:
            metadata = []
        
        if not metadata or len(metadata) == 0:
            # 如果没有元数据，假设顺序一致
            return timeline_idx
        
        # 尝试根据路径或时间戳匹配
        seq = int(row['seq'])
        timestamp = row.get('average', row.get('lidar_360', 0))
        
        # 方法1: 根据路径和时间戳匹配（适用于检测后的特征）
        for i, meta in enumerate(metadata):
            meta_path = meta.get('path', '')
            meta_name = meta.get('name', '')
            
            # 检查序列号是否匹配
            if f'seq{seq}' in meta_path or f'seq{seq:04d}' in meta_path or f'/{seq}/' in meta_path:
                # 检查时间戳是否接近
                # 对于检测后的特征，文件名可能包含：
                # 1. 时间戳范围格式: 170001.123_170001.456_cluster001_0000.npy
                # 2. 序列+时间戳格式: seq0001_1706255121.741490.jpg
                # 3. 单个时间戳格式: 170001.123.jpg
                try:
                    # 尝试从文件名提取时间戳
                    name_stem = Path(meta_name).stem
                    meta_timestamp = None
                    
                    if '_' in name_stem:
                        # 分割文件名
                        parts = name_stem.split('_')
                        
                        # 尝试找到时间戳部分（数字字符串，可能包含小数点）
                        # 格式1: seq0001_1706255121.741490 -> parts[1]是时间戳
                        # 格式2: 170001.123_170001.456_... -> parts[0]和parts[1]都是时间戳
                        for part in parts:
                            try:
                                # 尝试将part转换为float，如果是时间戳应该能成功
                                candidate = float(part)
                                # 检查是否是合理的时间戳（大于某个阈值，比如1000000000，即2001年）
                                if candidate > 1000000000.0:
                                    meta_timestamp = candidate
                                    break
                            except ValueError:
                                continue
                    else:
                        # 没有下划线，直接尝试整个stem作为时间戳
                        try:
                            meta_timestamp = float(name_stem)
                        except ValueError:
                            pass
                    
                    # 如果成功提取时间戳，检查是否匹配
                    if meta_timestamp is not None:
                        # 检查时间戳是否在窗口内（检测结果可能对应时间范围）
                        if abs(meta_timestamp - timestamp) < 0.5:  # 0.5秒容差（检测窗口可能较大）
                            return i
                except (ValueError, AttributeError, TypeError):
                    # 如果无法解析时间戳，继续尝试其他方法
                    pass
        
        # 方法2: 如果特征数量与时间线一致，直接使用索引
        if feature_type == "pointnext" or feature_type == "livox":
            num_features = len(self.livox_features) if hasattr(self, 'livox_features') and self.livox_features is not None else (len(self.pointnext_features) if self.pointnext_features is not None else 0)
        elif feature_type == "convnext" or feature_type == "image":
            num_features = len(self.image_features) if hasattr(self, 'image_features') and self.image_features is not None else (len(self.convnext_features) if self.convnext_features is not None else 0)
        elif feature_type == "lidar360":
            num_features = len(self.lidar360_features) if self.lidar360_features is not None else 0
        elif feature_type == "radar":
            num_features = len(self.radar_features) if hasattr(self, 'radar_features') and self.radar_features is not None else 0
        else:
            num_features = 0
        
        if num_features == len(self.timeline):
            return timeline_idx
        
        # 方法3: 返回None，表示未找到匹配
        return None

    def _build_class_file_cache(self):
        """预加载所有类别文件的时间戳索引（提高匹配效率，解决时间戳精度问题）。"""
        self.class_file_cache = {}  # (seq, timestamp) -> file_path
        
        for seq_dir in self.base_dir.glob("seq*"):
            if not seq_dir.is_dir():
                continue
            try:
                seq = int(seq_dir.name.replace('seq', ''))
                class_dir = seq_dir / "class"
                if class_dir.exists():
                    for class_file in class_dir.glob("*.npy"):
                        try:
                            file_timestamp = float(class_file.stem)
                            self.class_file_cache[(seq, file_timestamp)] = class_file
                        except ValueError:
                            continue
            except ValueError:
                continue

    def _aggregate_features(
        self,
        features: torch.Tensor,
        mode: str
    ) -> torch.Tensor:
        """
        聚合特征。
        
        Args:
            features: 特征张量，形状 (N, D)
            mode: 聚合模式
        
        Returns:
            聚合后的特征，形状 (D,) 或 (N*D,)
        """
        if len(features) == 0:
            return torch.zeros(features.shape[1] if len(features.shape) > 1 else 1)
        
        if mode == "mean":
            return features.mean(dim=0)
        elif mode == "max":
            return features.max(dim=0)[0]
        elif mode == "attention":
            # 简单的注意力聚合（可以扩展为可学习的注意力）
            weights = torch.softmax(torch.ones(len(features)), dim=0)
            return (features * weights.unsqueeze(1)).sum(dim=0)
        elif mode == "concat":
            return features.flatten()
        else:
            raise ValueError(f"未知的聚合模式: {mode}")

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个时间窗口的样本（四模态：图像、Livox、Lidar 360、Radar）。
        
        Returns:
            {
                'image_feat': 聚合后的图像特征（ConvNeXt），
                'livox_feat': 聚合后的Livox点云特征（PointNeXt），
                'lidar360_feat': 聚合后的Lidar 360点云特征（PointNeXt），
                'radar_feat': 聚合后的Radar点云特征（PointNeXt），
                'image_mask': 图像特征存在掩码（1表示存在，0表示缺失），
                'livox_mask': Livox特征存在掩码，
                'lidar360_mask': Lidar 360特征存在掩码，
                'radar_mask': Radar特征存在掩码，
                'image_conf': 图像检测置信度（0-1），
                'livox_conf': Livox检测置信度（0-1），
                'lidar360_conf': Lidar 360检测置信度（0-1），
                'radar_conf': Radar检测置信度（0-1），
                'image_point_count': 图像检测框数量（用于权重计算），
                'livox_point_count': Livox点云数量（用于权重计算），
                'lidar360_point_count': Lidar 360点云数量（用于权重计算），
                'radar_point_count': Radar点云数量（用于权重计算），
                'label': 类别标签（one-hot编码），
                'center_time': 中心时间戳,
                'seq': 序列号,
                # 向后兼容的旧字段
                'convnext_feat': 等同于image_feat,
                'pointnext_feat': 等同于livox_feat,
                'fused_feat': 拼接后的融合特征（向后兼容），
            }
        """
        window_info = self.window_indices[idx]
        
        # 检查是否直接使用特征索引（使用预计算特征且特征文件已经划分好）
        use_feature_idx_directly = window_info.get('use_feature_idx_directly', False)
        
        if use_feature_idx_directly:
            # 直接使用特征索引（不通过timeline）
            feature_idx = window_info.get('feature_idx', window_info['center_idx'])
            
            # 从元数据获取信息
            seq = None
            center_time = window_info.get('center_time', 0.0)
            
            # 尝试从元数据获取seq（优先使用image_metadata）
            if self.image_metadata and feature_idx < len(self.image_metadata):
                meta = self.image_metadata[feature_idx]
                if isinstance(meta, dict):
                    seq = meta.get('seq')
                    center_time = meta.get('window_center', meta.get('timestamp', center_time))
            elif self.livox_metadata and feature_idx < len(self.livox_metadata):
                meta = self.livox_metadata[feature_idx]
                if isinstance(meta, dict):
                    seq = meta.get('seq')
                    center_time = meta.get('window_center', meta.get('timestamp', center_time))
            
            if seq is None:
                seq = 0  # 默认值
            
            # 直接使用特征索引获取特征
            image_feats = []
            livox_feats = []
            lidar360_feats = []
            radar_feats = []
            image_confs = []
            livox_confs = []
            lidar360_confs = []
            radar_confs = []
            image_point_counts = []
            livox_point_counts = []
            lidar360_point_counts = []
            radar_point_counts = []
            
            # 获取图像特征
            if self.image_features is not None and feature_idx < len(self.image_features):
                feat = self.image_features[feature_idx]
                if feat.norm() > 1e-6:
                    image_feats.append(feat)
                    conf = 1.0
                    if feature_idx < len(self.image_metadata) and isinstance(self.image_metadata[feature_idx], dict):
                        conf = self.image_metadata[feature_idx].get('confidence', 1.0)
                        image_point_counts.append(self.image_metadata[feature_idx].get('point_count', 0))
                    image_confs.append(conf)
            
            # 获取Livox特征
            if self.livox_features is not None and feature_idx < len(self.livox_features):
                feat = self.livox_features[feature_idx]
                if feat.norm() > 1e-6:
                    livox_feats.append(feat)
                    conf = 1.0
                    if feature_idx < len(self.livox_metadata) and isinstance(self.livox_metadata[feature_idx], dict):
                        conf = self.livox_metadata[feature_idx].get('confidence', 1.0)
                        livox_point_counts.append(self.livox_metadata[feature_idx].get('point_count', 0))
                    livox_confs.append(conf)
            
            # 获取Lidar 360特征
            if self.lidar360_features is not None and feature_idx < len(self.lidar360_features):
                feat = self.lidar360_features[feature_idx]
                if feat.norm() > 1e-6:
                    lidar360_feats.append(feat)
                    conf = 1.0
                    if feature_idx < len(self.lidar360_metadata) and isinstance(self.lidar360_metadata[feature_idx], dict):
                        conf = self.lidar360_metadata[feature_idx].get('confidence', 1.0)
                        lidar360_point_counts.append(self.lidar360_metadata[feature_idx].get('point_count', 0))
                    lidar360_confs.append(conf)
            
            # 获取Radar特征
            if self.radar_features is not None and feature_idx < len(self.radar_features):
                feat = self.radar_features[feature_idx]
                if feat.norm() > 1e-6:
                    radar_feats.append(feat)
                    conf = 1.0
                    if feature_idx < len(self.radar_metadata) and isinstance(self.radar_metadata[feature_idx], dict):
                        conf = self.radar_metadata[feature_idx].get('confidence', 1.0)
                        radar_point_counts.append(self.radar_metadata[feature_idx].get('point_count', 0))
                    radar_confs.append(conf)
        else:
            # 使用timeline索引（原有逻辑）
            center_idx = window_info['center_idx']
            window_indices = window_info['window_indices']
            window_indices_for_concat = window_indices  # 用于concat模式
            
            # 获取中心样本的信息
            center_row = self.timeline.iloc[center_idx]
            seq = int(center_row['seq'])
            center_time = window_info.get('center_time', 0.0)
            
            # 收集窗口内的特征（四种模态）
            image_feats = []
            livox_feats = []
            lidar360_feats = []
            radar_feats = []
            
            # 收集置信度（从元数据中提取）
            image_confs = []
            livox_confs = []
            lidar360_confs = []
            radar_confs = []
            
            # 收集点云数量（用于权重计算）
            image_point_counts = []  # 图像检测框数量
            livox_point_counts = []  # Livox点云数量
            lidar360_point_counts = []  # Lidar 360点云数量
            radar_point_counts = []  # Radar点云数量
            
            for win_idx in window_indices:
                row = self.timeline.iloc[win_idx]
                
                # 获取图像特征（ConvNeXt）
                if self.use_precomputed_features and self.image_features is not None:
                    feat_idx = self._match_feature_index(win_idx, row, "image")
                    if feat_idx is not None and feat_idx < len(self.image_features):
                        feat = self.image_features[feat_idx]
                        if feat.norm() > 1e-6:  # 非零特征
                            image_feats.append(feat)
                            # 从元数据提取置信度
                            conf = 1.0
                            if feat_idx < len(self.image_metadata) and isinstance(self.image_metadata[feat_idx], dict):
                                conf = self.image_metadata[feat_idx].get('confidence', 1.0)
                            image_confs.append(conf)
                elif self.use_precomputed_features and self.convnext_features is not None:
                    # 向后兼容
                    feat_idx = self._match_feature_index(win_idx, row, "convnext")
                    if feat_idx is not None and feat_idx < len(self.convnext_features):
                        feat = self.convnext_features[feat_idx]
                        if feat.norm() > 1e-6:
                            image_feats.append(feat)
                            image_confs.append(1.0)
            
            # 获取Livox点云特征
            if self.use_precomputed_features and self.livox_features is not None:
                feat_idx = self._match_feature_index(win_idx, row, "livox")
                if feat_idx is not None and feat_idx < len(self.livox_features):
                    feat = self.livox_features[feat_idx]
                    if feat.norm() > 1e-6:
                        livox_feats.append(feat)
                        conf = 1.0
                        if feat_idx < len(self.livox_metadata) and isinstance(self.livox_metadata[feat_idx], dict):
                            conf = self.livox_metadata[feat_idx].get('confidence', 1.0)
                        livox_confs.append(conf)
            elif self.use_precomputed_features and self.pointnext_features is not None:
                # 向后兼容
                feat_idx = self._match_feature_index(win_idx, row, "pointnext")
                if feat_idx is not None and feat_idx < len(self.pointnext_features):
                    feat = self.pointnext_features[feat_idx]
                    if feat.norm() > 1e-6:
                        livox_feats.append(feat)
                        livox_confs.append(1.0)
            
            # 获取Lidar 360点云特征
            if self.use_precomputed_features and self.lidar360_features is not None:
                feat_idx = self._match_feature_index(win_idx, row, "lidar360")
                if feat_idx is not None and feat_idx < len(self.lidar360_features):
                    feat = self.lidar360_features[feat_idx]
                    if feat.norm() > 1e-6:
                        lidar360_feats.append(feat)
                        conf = 1.0
                        point_count = 0
                        if feat_idx < len(self.lidar360_metadata) and isinstance(self.lidar360_metadata[feat_idx], dict):
                            conf = self.lidar360_metadata[feat_idx].get('confidence', 1.0)
                            point_count = self.lidar360_metadata[feat_idx].get('point_count', 0)
                        lidar360_confs.append(conf)
                        lidar360_point_counts.append(point_count)
            
            # 获取Radar点云特征
            if self.use_precomputed_features and self.radar_features is not None:
                feat_idx = self._match_feature_index(win_idx, row, "radar")
                if feat_idx is not None and feat_idx < len(self.radar_features):
                    feat = self.radar_features[feat_idx]
                    if feat.norm() > 1e-6:
                        radar_feats.append(feat)
                        conf = 1.0
                        point_count = 0
                        if feat_idx < len(self.radar_metadata) and isinstance(self.radar_metadata[feat_idx], dict):
                            conf = self.radar_metadata[feat_idx].get('confidence', 1.0)
                            point_count = self.radar_metadata[feat_idx].get('point_count', 0)
                        radar_confs.append(conf)
                        radar_point_counts.append(point_count)
        
        # 聚合特征
        # 图像特征
        if image_feats:
            image_tensor = torch.stack(image_feats)
            image_agg = self._aggregate_features(image_tensor, self.aggregation_mode)
            image_mask = torch.tensor(1.0, dtype=torch.float32)
            image_conf = torch.tensor(np.mean(image_confs) if image_confs else 1.0, dtype=torch.float32)
        else:
            feat_dim = self.image_features.shape[1] if (hasattr(self, 'image_features') and self.image_features is not None) else (self.convnext_features.shape[1] if self.convnext_features is not None else 768)
            if self.aggregation_mode == "concat":
                image_agg = torch.zeros(feat_dim * len(window_indices_for_concat), dtype=torch.float32)
            else:
                image_agg = torch.zeros(feat_dim, dtype=torch.float32)
            image_mask = torch.tensor(0.0, dtype=torch.float32)
            image_conf = torch.tensor(0.0, dtype=torch.float32)
        
        # Livox特征
        if livox_feats:
            livox_tensor = torch.stack(livox_feats)
            livox_agg = self._aggregate_features(livox_tensor, self.aggregation_mode)
            livox_mask = torch.tensor(1.0, dtype=torch.float32)
            livox_conf = torch.tensor(np.mean(livox_confs) if livox_confs else 1.0, dtype=torch.float32)
        else:
            feat_dim = self.livox_features.shape[1] if (hasattr(self, 'livox_features') and self.livox_features is not None) else (self.pointnext_features.shape[1] if self.pointnext_features is not None else 512)
            if self.aggregation_mode == "concat":
                livox_agg = torch.zeros(feat_dim * len(window_indices_for_concat), dtype=torch.float32)
            else:
                livox_agg = torch.zeros(feat_dim, dtype=torch.float32)
            livox_mask = torch.tensor(0.0, dtype=torch.float32)
            livox_conf = torch.tensor(0.0, dtype=torch.float32)
        
        # Lidar 360特征
        if lidar360_feats:
            lidar360_tensor = torch.stack(lidar360_feats)
            lidar360_agg = self._aggregate_features(lidar360_tensor, self.aggregation_mode)
            lidar360_mask = torch.tensor(1.0, dtype=torch.float32)
            lidar360_conf = torch.tensor(np.mean(lidar360_confs) if lidar360_confs else 1.0, dtype=torch.float32)
            lidar360_point_count = torch.tensor(np.sum(lidar360_point_counts) if lidar360_point_counts else 0, dtype=torch.float32)
        else:
            feat_dim = self.lidar360_features.shape[1] if (hasattr(self, 'lidar360_features') and self.lidar360_features is not None) else 512
            if self.aggregation_mode == "concat":
                lidar360_agg = torch.zeros(feat_dim * len(window_indices_for_concat), dtype=torch.float32)
            else:
                lidar360_agg = torch.zeros(feat_dim, dtype=torch.float32)
            lidar360_mask = torch.tensor(0.0, dtype=torch.float32)
            lidar360_conf = torch.tensor(0.0, dtype=torch.float32)
            lidar360_point_count = torch.tensor(0.0, dtype=torch.float32)
        
        # Radar特征
        if radar_feats:
            radar_tensor = torch.stack(radar_feats)
            radar_agg = self._aggregate_features(radar_tensor, self.aggregation_mode)
            radar_mask = torch.tensor(1.0, dtype=torch.float32)
            radar_conf = torch.tensor(np.mean(radar_confs) if radar_confs else 1.0, dtype=torch.float32)
            radar_point_count = torch.tensor(np.sum(radar_point_counts) if radar_point_counts else 0, dtype=torch.float32)
        else:
            feat_dim = self.radar_features.shape[1] if (hasattr(self, 'radar_features') and self.radar_features is not None) else 512
            if self.aggregation_mode == "concat":
                radar_agg = torch.zeros(feat_dim * len(window_indices_for_concat), dtype=torch.float32)
            else:
                radar_agg = torch.zeros(feat_dim, dtype=torch.float32)
            radar_mask = torch.tensor(0.0, dtype=torch.float32)
            radar_conf = torch.tensor(0.0, dtype=torch.float32)
            radar_point_count = torch.tensor(0.0, dtype=torch.float32)
        
        # 归一化
        if self.normalize_features:
            if image_agg.norm() > 0:
                image_agg = torch.nn.functional.normalize(image_agg, p=2, dim=0)
            if livox_agg.norm() > 0:
                livox_agg = torch.nn.functional.normalize(livox_agg, p=2, dim=0)
            if lidar360_agg.norm() > 0:
                lidar360_agg = torch.nn.functional.normalize(lidar360_agg, p=2, dim=0)
            if radar_agg.norm() > 0:
                radar_agg = torch.nn.functional.normalize(radar_agg, p=2, dim=0)
        
        # 加载类别标签（支持4类分类：类别0, 1, 2, 3）
        # 优先从元数据中获取类别ID（如果使用重组特征）
        class_id = -1
        
        # 如果直接使用特征索引，从元数据获取类别
        if use_feature_idx_directly:
            # 优先从image_metadata获取（因为它通常包含最完整的信息）
            if self.image_metadata and feature_idx < len(self.image_metadata):
                meta = self.image_metadata[feature_idx]
                if isinstance(meta, dict) and 'class_id' in meta:
                    class_id = int(meta['class_id'])
            # 如果image_metadata没有，尝试其他元数据
            elif self.livox_metadata and feature_idx < len(self.livox_metadata):
                meta = self.livox_metadata[feature_idx]
                if isinstance(meta, dict) and 'class_id' in meta:
                    class_id = int(meta['class_id'])
            elif self.lidar360_metadata and feature_idx < len(self.lidar360_metadata):
                meta = self.lidar360_metadata[feature_idx]
                if isinstance(meta, dict) and 'class_id' in meta:
                    class_id = int(meta['class_id'])
            elif self.radar_metadata and feature_idx < len(self.radar_metadata):
                meta = self.radar_metadata[feature_idx]
                if isinstance(meta, dict) and 'class_id' in meta:
                    class_id = int(meta['class_id'])
        
        # 如果元数据中没有类别，则从文件加载
        if class_id == -1:
            # 使用中心时间戳查找类别文件
            if use_feature_idx_directly:
                center_timestamp = center_time
                class_timestamp = center_time
            else:
                center_timestamp = center_row['average']
                class_timestamp = center_row.get('class', center_timestamp)
            
            # 方法1: 精确匹配
            if pd.notna(class_timestamp):
                class_path = self.base_dir / f"seq{seq:04d}" / "class" / f"{class_timestamp:.7f}.npy"
                if class_path.exists():
                    try:
                        class_id = int(np.load(class_path).item() if hasattr(np.load(class_path), 'item') else np.load(class_path))
                    except Exception:
                        pass
            
            # 方法2: 使用预加载的缓存进行容差匹配（0.1秒容差，解决时间戳精度问题）
            if class_id == -1 and self.class_file_cache:
                best_match = None
                best_diff = float('inf')
                time_tolerance = 0.1  # 0.1秒容差
                
                for (s, file_ts), file_path in self.class_file_cache.items():
                    if s == seq:
                        diff = abs(file_ts - class_timestamp)
                        if diff < best_diff and diff <= time_tolerance:
                            best_diff = diff
                            best_match = file_path
                
                if best_match and best_match.exists():
                    try:
                        class_id = int(np.load(best_match).item() if hasattr(np.load(best_match), 'item') else np.load(best_match))
                    except Exception:
                        pass
        
        # 创建 one-hot 编码
        label = torch.zeros(4, dtype=torch.float32)  # 4类分类
        if 0 <= class_id <= 3:
            label[class_id] = 1.0  # one-hot编码（类别0对应索引0，类别1对应索引1，以此类推）
        else:
            # 如果无法加载类别，默认使用类别0（但应该打印警告）
            # label[0] = 1.0  # 注释掉，避免隐藏问题
            pass
        
        # 计算点云数量（图像使用检测框数量）
        image_point_count = torch.tensor(len(image_feats) if image_feats else 0, dtype=torch.float32)
        livox_point_count = torch.tensor(np.sum(livox_point_counts) if livox_point_counts else 0, dtype=torch.float32)
        
        # 构建返回字典
        result = {
            # 新字段（四模态）
            'image_feat': image_agg,
            'livox_feat': livox_agg,
            'lidar360_feat': lidar360_agg,
            'radar_feat': radar_agg,
            'image_mask': image_mask,
            'livox_mask': livox_mask,
            'lidar360_mask': lidar360_mask,
            'radar_mask': radar_mask,
            'image_conf': image_conf,
            'livox_conf': livox_conf,
            'lidar360_conf': lidar360_conf,
            'radar_conf': radar_conf,
            'image_point_count': image_point_count,
            'livox_point_count': livox_point_count,
            'lidar360_point_count': lidar360_point_count,
            'radar_point_count': radar_point_count,
            # 向后兼容的旧字段
            'convnext_feat': image_agg,
            'pointnext_feat': livox_agg,
            'fused_feat': torch.cat([livox_agg, image_agg], dim=0),  # 向后兼容
            # 其他字段
            'label': label,  # label已经是float32
            'center_time': torch.tensor(window_info['center_time'], dtype=torch.float32),
            'seq': torch.tensor(seq, dtype=torch.long),
        }
        return result


def _load_timeline(timeline_dir: Path) -> pd.DataFrame:
    """兼容旧接口的辅助函数。"""
    result = None
    for file_path in sorted(timeline_dir.iterdir(), key=lambda x: int(x.stem) if x.stem.isdigit() else 0):
        if result is None:
            result = pd.read_csv(file_path, delimiter='\t', dtype='str')
        else:
            result = pd.concat((result, pd.read_csv(file_path, delimiter='\t', dtype='str')))
    return result.reset_index()

