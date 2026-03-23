import os
import sys
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import DBSCAN

# 允许直接运行脚本时也能导入 point_cloud_processing 包
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from point_cloud_processing.training_lidar.dataset_loader import read_lidar_files  # noqa: E402
from point_cloud_processing.training_lidar.extract_feature import extract_feature_set_predict  # noqa: E402
from point_cloud_processing.training_lidar.train_radar_detector import RadarLSTMClassifier  # noqa: E402


def _timestamp_sort_key(timestamp: str):
    try:
        return float(timestamp)
    except ValueError:
        return timestamp


def _prepare_frame_groups(
    radar_frames: Dict[str, np.ndarray],
    window_size: int,
) -> List[Tuple[Sequence[str], np.ndarray]]:
    """
    将雷达点云按时间窗口累积（默认 4 帧，非重叠窗口，步长=window_size）。
    与训练特征构建保持一致：空帧过滤后仍保留时间维度，使用帧索引。
    """
    if not radar_frames:
        return []

    point_dim = None
    for frame in radar_frames.values():
        if frame.size:
            # 处理一维数组（shape为(N,)的情况）
            if frame.ndim == 1:
                # 如果是一维数组，假设是单个点（3个坐标）
                if frame.shape[0] >= 3:
                    point_dim = 3
                else:
                    point_dim = frame.shape[0]
            else:
                point_dim = frame.shape[1]
            break
    if point_dim is None:
        # 即使所有帧为空，也保证输出维度正确
        one = next(iter(radar_frames.values()))
        if one.ndim == 1:
            point_dim = one.shape[0] if one.size >= 3 else 3
        else:
            point_dim = one.shape[1] if one.size else 3

    sorted_items = sorted(radar_frames.items(), key=lambda item: _timestamp_sort_key(item[0]))
    groups: List[Tuple[Sequence[str], np.ndarray]] = []

    current_timestamps: List[str] = []
    current_batches: List[np.ndarray] = []

    for timestamp, raw_points in sorted_items:
        current_timestamps.append(timestamp)

        # 确保raw_points是二维数组
        if raw_points.ndim == 1:
            # 如果是一维数组，reshape为(1, N)
            raw_points = raw_points.reshape(1, -1)

        # 过滤零值数据
        mask = np.any(raw_points != 0, axis=1)
        filtered = raw_points[mask]

        # 帧索引按窗口内位置 1..window_size
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
            # 非重叠窗口：重置缓冲，步长=window_size
            current_timestamps = []
            current_batches = []

    return groups


@dataclass
class RadarDetectionSample:
    """
    雷达检测输出的单个样本，包含累积四帧的目标点云聚类。
    """
    sequence_name: str
    timestamps: Tuple[str, ...]
    points: np.ndarray  # shape: (N, point_dim + 1) -> [frame_idx, xyz{, ...}]
    score: float
    cluster_id: int


class RadarCenterSequenceDetector:
    """
    基于雷达点云聚类的 LSTM 二分类检测器。

    该检测器与 ``train_radar_detector.py`` 中的训练脚本保持一致，使用四帧窗口检测（window_size=4），
    对累积的4帧点云进行DBSCAN聚类，提取每个聚类的特征（位置、速度、加速度、标准差、范围，共 15 维），
    判断每个聚类是否存在无人机目标。
    """

    def __init__(
        self,
        model_path: str,
        *,
        hidden_size: int = 64,  # 与train_radar_detector.py默认值一致
        num_layers: int = 1,    # 与train_radar_detector.py默认值一致
        dropout: float = 0.6,   # 与train_radar_detector.py默认值一致
        window_size: int = 4,
        dbscan_eps: float = 1.0,  # 与训练时一致
        dbscan_min_samples: int = 3,  # 与训练时一致
        prob_threshold: float = 0.6,
        scaler_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.window_size = window_size
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.prob_threshold = prob_threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 加载模型文件（支持完整checkpoint和直接state_dict两种格式）
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 检查是否是完整的checkpoint（包含model_state_dict键）
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 完整checkpoint格式（train_radar_detector.py保存的格式）
            state_dict = checkpoint['model_state_dict']
            # 从checkpoint中获取模型配置（优先使用checkpoint中的配置）
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
                inferred_input_size = model_config.get('input_size', 15)
                inferred_hidden_size = model_config.get('hidden_size', hidden_size)
                inferred_num_layers = model_config.get('num_layers', num_layers)
                inferred_dropout = model_config.get('dropout_rate', model_config.get('dropout', dropout))
                print(f"✓ 从checkpoint加载模型配置:")
                print(f"  input_size={inferred_input_size}, hidden_size={inferred_hidden_size}, num_layers={inferred_num_layers}, dropout={inferred_dropout}")
                # 使用checkpoint中的配置
                hidden_size = inferred_hidden_size
                num_layers = inferred_num_layers
                dropout = inferred_dropout
            else:
                # checkpoint中没有model_config，使用默认值
                inferred_input_size = 15  # 雷达特征默认15维
                print(f"⚠ checkpoint中未找到model_config，使用默认配置:")
                print(f"  input_size={inferred_input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
        else:
            # 直接state_dict格式（向后兼容）
            state_dict = checkpoint
            inferred_input_size = 15  # 雷达特征默认15维
            print(f"⚠ 加载直接state_dict格式，使用参数配置:")
            print(f"  input_size={inferred_input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
        
        self.model = RadarLSTMClassifier(
            input_size=inferred_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_rate=dropout,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✓ 模型加载完成")
        
        # 加载特征标准化器（scaler）
        self.scaler = None
        if scaler_path is None:
            # 尝试从模型路径自动推断scaler路径
            scaler_path = model_path.replace('.pth', '_scaler.pkl')
            if not os.path.exists(scaler_path):
                # 尝试其他可能的路径
                alt_path = model_path.replace('lstm_radar_enhance_model.pth', 'lstm_radar_enhance_model_optimized_scaler.pkl')
                if os.path.exists(alt_path):
                    scaler_path = alt_path
                else:
                    scaler_path = None
        
        if scaler_path and os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✓ 加载特征标准化器: {scaler_path}")
            except Exception as e:
                print(f"⚠ 加载scaler失败: {e}，将使用未标准化的特征（可能影响性能）")
        else:
            print(f"⚠ 未找到scaler文件，将使用未标准化的特征（可能影响性能）")
            if scaler_path:
                print(f"  尝试的路径: {scaler_path}")

    def _classify_clusters(
        self,
        accumulated_points: np.ndarray,
        timestamp_group: Sequence[str],
    ) -> List[RadarDetectionSample]:
        """
        对单个时间窗口执行聚类并筛选出正样本聚类。
        """
        if accumulated_points.size == 0:
            return []

        time_indices = accumulated_points[:, 0]
        xyz_points = accumulated_points[:, 1:4]  # 只使用xyz坐标

        if xyz_points.shape[0] < self.dbscan_min_samples:
            return []

        # 根据点数动态调整 min_samples，避免小目标被过滤
        dynamic_min_samples = 1 if xyz_points.shape[0] < 5 else self.dbscan_min_samples
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=dynamic_min_samples)
        labels = clustering.fit_predict(xyz_points)

        unique_labels = [label for label in np.unique(labels) if label != -1]
        if not unique_labels:
            return []

        feature_set, cluster_ids = extract_feature_set_predict(xyz_points, labels, time_indices)
        if feature_set.size == 0:
            return []

        # 如果时间维度超过预期，截断到4帧（取前4帧）
        expected_time_dim = self.window_size  # 应该是4帧
        current_time_dim = feature_set.shape[1]
        
        if current_time_dim > expected_time_dim:
            feature_set = feature_set[:, :expected_time_dim, :]

        # 特征提取返回15维（mean(3) + std(3) + range(3) + velocity(3) + acceleration(3)）
        # feature_set 的形状是 [num_clusters, num_timesteps, feature_dim]
        # 获取模型的输入维度
        model_input_size = self.model.lstm.input_size
        original_feature_size = feature_set.shape[-1]
        if original_feature_size > model_input_size:
            # 特征维度大于模型输入维度，截断为前N维
            feature_set = feature_set[:, :, :model_input_size]
        elif original_feature_size < model_input_size:
            # 特征维度小于模型输入维度，填充零
            padding = np.zeros(
                (feature_set.shape[0], feature_set.shape[1], model_input_size - original_feature_size),
                dtype=feature_set.dtype
            )
            feature_set = np.concatenate([feature_set, padding], axis=-1)

        # 特征标准化（如果scaler存在）
        if self.scaler is not None:
            batch_shape = feature_set.shape
            batch_flat = feature_set.reshape(-1, batch_shape[-1])
            batch_scaled = self.scaler.transform(batch_flat)
            feature_set = batch_scaled.reshape(batch_shape)

        # 转换为tensor并分类
        inputs = torch.tensor(feature_set, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)
            positive_probs = probs[:, 1].detach().cpu().numpy()

        # 筛选正样本聚类
        samples: List[RadarDetectionSample] = []
        for cluster_idx, prob in enumerate(positive_probs):
            if prob < self.prob_threshold:
                continue

            cluster_label = cluster_ids[cluster_idx]
            cluster_mask = labels == cluster_label
            cluster_points = accumulated_points[cluster_mask]

            samples.append(
                RadarDetectionSample(
                    sequence_name="",  # 将在调用处设置
                    timestamps=timestamp_group,
                    points=cluster_points.copy(),
                    score=float(prob),
                    cluster_id=int(cluster_label),
                )
            )

        return samples

    def detect_from_frames(
        self,
        sequence_name: str,
        radar_frames: Dict[str, np.ndarray],
    ) -> List[RadarDetectionSample]:
        """
        从雷达点云帧字典中检测目标。
        """
        frame_groups = _prepare_frame_groups(radar_frames, self.window_size)
        all_samples: List[RadarDetectionSample] = []

        for timestamp_group, accumulated_points in frame_groups:
            samples = self._classify_clusters(accumulated_points, timestamp_group)
            for sample in samples:
                sample.sequence_name = sequence_name
            all_samples.extend(samples)

        return all_samples

    def detect_sequence_directory(
        self,
        sequence_path: str,
        *,
        radar_subdir: str = "radar_enhance_pcl",
    ) -> List[RadarDetectionSample]:
        radar_dir = os.path.join(sequence_path, radar_subdir)
        if not os.path.isdir(radar_dir):
            raise FileNotFoundError(f"未找到雷达点云目录: {radar_dir}")

        sequence_name = os.path.basename(sequence_path.rstrip("/"))
        radar_frames = read_lidar_files(radar_dir)
        return self.detect_from_frames(sequence_name, radar_frames)

    def detect_dataset(
        self,
        dataset_folder: str,
        *,
        radar_subdir: str = "radar_enhance_pcl",
        output_subdir: Optional[str] = None,
        metadata_filename: str = "radar_detections.csv",
    ) -> Dict[str, List[RadarDetectionSample]]:
        if not os.path.isdir(dataset_folder):
            raise FileNotFoundError(f"数据集目录不存在: {dataset_folder}")

        all_results: Dict[str, List[RadarDetectionSample]] = {}
        metadata_rows: List[str] = [
            "sequence_name,timestamp_start,timestamp_end,score,points_path,cluster_id"
        ]

        for sequence_name in sorted(os.listdir(dataset_folder)):
            sequence_path = os.path.join(dataset_folder, sequence_name)
            if not os.path.isdir(sequence_path):
                continue

            try:
                samples = self.detect_sequence_directory(
                    sequence_path,
                    radar_subdir=radar_subdir,
                )
            except FileNotFoundError:
                continue

            all_results[sequence_name] = samples

            if output_subdir is None or not samples:
                continue

            output_dir = os.path.join(sequence_path, output_subdir)
            os.makedirs(output_dir, exist_ok=True)

            for index, sample in enumerate(samples):
                timestamp_start = sample.timestamps[0]
                timestamp_end = sample.timestamps[-1]
                filename = f"{timestamp_start}_{timestamp_end}_radar_cluster{sample.cluster_id}_{index:04d}.npy"
                output_path = os.path.join(output_dir, filename)
                np.save(output_path, sample.points.astype(np.float32))

                metadata_rows.append(
                    ",".join(
                        [
                            sequence_name,
                            timestamp_start,
                            timestamp_end,
                            f"{sample.score:.6f}",
                            os.path.relpath(output_path, dataset_folder),
                            str(sample.cluster_id),
                        ]
                    )
                )

        if output_subdir is not None and len(metadata_rows) > 1:
            metadata_path = os.path.join(dataset_folder, metadata_filename)
            with open(metadata_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(metadata_rows))

        return all_results


__all__ = [
    "RadarCenterSequenceDetector",
    "RadarDetectionSample",
]
