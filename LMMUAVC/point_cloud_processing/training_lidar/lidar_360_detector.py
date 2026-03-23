"""
Lidar 360 点云目标检测器。
基于 LivoxAviaDetector，累积4帧（与 Livox Avia 相同）。
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

# 允许直接运行脚本时也能导入 point_cloud_processing 包
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from point_cloud_processing.training_lidar.dataset_loader import (  # noqa: E402
    read_lidar_files,
)
from point_cloud_processing.training_lidar.extract_feature import (  # noqa: E402
    extract_feature_set_predict,
)


class LSTMClassifier(nn.Module):
    """
    与训练脚本保持一致的 LSTM 分类器实现。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


@dataclass
class DetectionSample:
    """
    Lidar 360 检测输出的单个样本，包含累积四帧的目标点云。
    """

    sequence_name: str
    timestamps: Tuple[str, ...]
    points: np.ndarray  # shape: (N, point_dim + 1) -> [frame_idx, xyz{, ...}]
    score: float
    cluster_id: int


def _farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    最远点采样，用于裁剪点云数量。
    """
    if n_samples <= 0 or points.shape[0] <= n_samples:
        return points

    xyz = points[:, 1:4] if points.shape[1] > 3 else points[:, :3]  # 忽略时间索引
    centroids = np.zeros((n_samples,), dtype=np.int32)
    distance = np.ones((points.shape[0],), dtype=np.float64) * 1e10
    farthest = np.random.randint(0, points.shape[0])

    for i in range(n_samples):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)

    return points[centroids]


def _prepare_frame_groups(
    lidar_frames: Dict[str, np.ndarray],
    window_size: int,
) -> List[Tuple[Sequence[str], np.ndarray]]:
    """
    将 Lidar 360 点云按时间窗口累积（默认 4 帧）。
    """
    if not lidar_frames:
        return []

    point_dim = None
    for frame in lidar_frames.values():
        if frame.size:
            point_dim = frame.shape[1]
            break
    if point_dim is None:
        # 即使所有帧为空，也保证输出维度正确
        one = next(iter(lidar_frames.values()))
        point_dim = one.shape[1] if one.size else 3

    sorted_items = sorted(lidar_frames.items(), key=lambda item: float(item[0]))
    groups: List[Tuple[Sequence[str], np.ndarray]] = []
    current_timestamps: List[str] = []
    current_batches: List[np.ndarray] = []

    for timestamp, raw_points in sorted_items:
        current_timestamps.append(timestamp)
        mask = np.any(raw_points != 0, axis=1)
        filtered = raw_points[mask]

        frame_index = len(current_timestamps) * np.ones((filtered.shape[0], 1), dtype=np.float32)
        if filtered.size == 0:
            data_with_index = np.empty((0, point_dim + 1), dtype=np.float32)
        else:
            filtered = filtered[:, :point_dim]
            data_with_index = np.concatenate((frame_index, filtered.astype(np.float32)), axis=1)

        current_batches.append(data_with_index)

        if len(current_timestamps) == window_size:
            concatenated = (
                np.concatenate(current_batches, axis=0)
                if current_batches
                else np.empty((0, point_dim + 1), dtype=np.float32)
            )
            groups.append((tuple(current_timestamps), concatenated))
            current_timestamps = []
            current_batches = []

    return groups


class Lidar360Detector:
    """
    Lidar 360 点云目标检测器。
    用于在推理阶段定位无人机目标，并输出累积四帧的点云子集。
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        *,
        window_size: int = 4,
        dbscan_eps: float = 1.5,
        dbscan_min_samples: int = 3,
        prob_threshold: float = 0.5,
        max_points: int = 1024,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.window_size = window_size
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.prob_threshold = prob_threshold
        self.max_points = max_points

        # 加载模型文件（支持完整checkpoint和直接state_dict两种格式）
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 检查是否是完整的checkpoint（包含model_state_dict键）
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整checkpoint格式
            state_dict = checkpoint['model_state_dict']
            # 尝试从checkpoint中获取模型配置
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                inferred_input_size = model_config.get('input_size', 15)
                inferred_hidden_size = model_config.get('hidden_size', 64)
                inferred_num_classes = model_config.get('num_classes', 2)
                inferred_dropout = model_config.get('dropout_rate', 0.5)
                print(f"从checkpoint加载模型配置: input_size={inferred_input_size}, hidden_size={inferred_hidden_size}, num_classes={inferred_num_classes}, dropout={inferred_dropout}")
            else:
                # 从state_dict推断
                if 'lstm.weight_ih_l0' in state_dict:
                    weight_shape = state_dict['lstm.weight_ih_l0'].shape
                    inferred_input_size = weight_shape[1]
                    inferred_hidden_size = weight_shape[0] // 4
                    print(f"从state_dict推断模型配置: input_size={inferred_input_size}, hidden_size={inferred_hidden_size}")
                else:
                    inferred_input_size = 15
                    inferred_hidden_size = 64
                    print(f"警告: 无法推断，使用默认值: input_size={inferred_input_size}")
                
                if 'fc.weight' in state_dict:
                    inferred_num_classes = state_dict['fc.weight'].shape[0]
                else:
                    inferred_num_classes = 2
                inferred_dropout = 0.5
        else:
            # 直接state_dict格式（向后兼容）
            state_dict = checkpoint
            inferred_input_size = 15
            inferred_hidden_size = 64
            inferred_num_classes = 2
            inferred_dropout = 0.5
            print(f"加载直接state_dict格式，使用默认配置: input_size={inferred_input_size}, hidden_size={inferred_hidden_size}")
        
        self.model = LSTMClassifier(
            input_size=inferred_input_size,
            hidden_size=inferred_hidden_size,
            num_layers=1,
            num_classes=inferred_num_classes,
            dropout_rate=inferred_dropout,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _classify_clusters(
        self,
        accumulated_points: np.ndarray,
        timestamp_group: Sequence[str],
    ) -> List[DetectionSample]:
        """
        对单个时间窗口执行聚类并筛选出正样本聚类。
        """
        if accumulated_points.size == 0:
            return []

        time_indices = accumulated_points[:, 0]
        xyz_points = accumulated_points[:, 1:]

        if xyz_points.shape[0] < self.dbscan_min_samples:
            return []

        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        labels = clustering.fit_predict(xyz_points)

        unique_labels = [label for label in np.unique(labels) if label != -1]
        if not unique_labels:
            return []

        feature_set, cluster_ids = extract_feature_set_predict(xyz_points, labels, time_indices)
        if feature_set.size == 0:
            return []

        inputs = torch.tensor(feature_set, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)

        positive_mask = (predictions == 1) & (probs[:, 1] >= self.prob_threshold)
        if not positive_mask.any():
            return []

        positive_cluster_ids = cluster_ids.squeeze(axis=1)[positive_mask.cpu().numpy()]
        positive_scores = probs[:, 1][positive_mask].cpu().numpy()

        samples: List[DetectionSample] = []
        for cluster_id, score in zip(positive_cluster_ids, positive_scores):
            cluster_mask = labels == cluster_id
            if not cluster_mask.any():
                continue

            cluster_points = accumulated_points[cluster_mask]
            cluster_points = _farthest_point_sample(cluster_points, self.max_points)

            samples.append(
                DetectionSample(
                    sequence_name="",
                    timestamps=tuple(timestamp_group),
                    points=cluster_points,
                    score=float(score),
                    cluster_id=int(cluster_id),
                )
            )

        return samples

    def detect_from_frames(
        self,
        sequence_name: str,
        lidar_frames: Dict[str, np.ndarray],
    ) -> List[DetectionSample]:
        """
        对单个序列的 Lidar 360 点云执行检测。
        返回每个检测出的无人机目标的点云子集（累积两帧）。
        """
        groups = _prepare_frame_groups(lidar_frames, self.window_size)
        if not groups:
            return []

        samples: List[DetectionSample] = []
        for timestamps, accumulated_points in groups:
            group_samples = self._classify_clusters(accumulated_points, timestamps)
            for sample in group_samples:
                sample.sequence_name = sequence_name
            samples.extend(group_samples)

        return samples

    def detect_sequence_directory(
        self,
        sequence_path: str,
        *,
        lidar_subdir: str = "lidar_360",
    ) -> List[DetectionSample]:
        """
        直接读取序列目录中的 Lidar 360 点云并执行检测。
        """
        lidar_dir = os.path.join(sequence_path, lidar_subdir)
        if not os.path.isdir(lidar_dir):
            raise FileNotFoundError(f"未找到 Lidar 360 点云目录: {lidar_dir}")

        sequence_name = os.path.basename(sequence_path.rstrip("/"))
        lidar_frames = read_lidar_files(lidar_dir)
        return self.detect_from_frames(sequence_name, lidar_frames)
