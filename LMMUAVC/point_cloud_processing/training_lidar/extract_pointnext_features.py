"""
使用 PointNeXt 对 Livox Avia 检测器生成的点云子集提取特征。

示例用法：
    python extract_pointnext_features.py \
        --point-cloud-dir /path/to/dataset/seq1/detections \
        --metadata-csv /path/to/dataset/detections_metadata.csv \
        --output /path/to/output/pointnext_features.pt \
        --cfg PointNeXt/cfgs/scanobjectnn/pointnext-s.yaml \
        --pretrained-path /path/to/pretrained/pointnext-s.pth
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# 添加 PointNeXt 到路径
POINTNEXT_ROOT = Path(__file__).parent.parent.parent / "PointNeXt"
if str(POINTNEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTNEXT_ROOT))

try:
    from openpoints.utils import EasyConfig  # type: ignore
    from openpoints.models import build_model_from_cfg  # type: ignore
    from openpoints.utils import load_checkpoint  # type: ignore
except ImportError as exc:
    raise ImportError(
        "未找到 openpoints，请确保 PointNeXt 已正确安装。"
        "运行: cd PointNeXt && source install.sh"
    ) from exc


class PointCloudDataset(Dataset):
    """从 .npy 文件加载点云数据的 Dataset。"""

    def __init__(
        self,
        point_cloud_paths: List[Path],
        metadata: Optional[pd.DataFrame] = None,
        num_points: int = 1024,
        normalize: bool = True,
    ):
        """
        Args:
            point_cloud_paths: 点云 .npy 文件路径列表
            metadata: 可选的元数据 DataFrame，包含序列名、时间戳等信息
            num_points: 采样点数，如果点云数量不足则重复采样
            normalize: 是否归一化点云坐标（中心化并缩放到单位球）
        """
        self.point_cloud_paths = point_cloud_paths
        self.metadata = metadata
        self.num_points = num_points
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.point_cloud_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.point_cloud_paths[idx]
        points = np.load(path).astype(np.float32)

        # 模型需要4通道输入（in_channels: 4）
        # 点云数据格式: [intensity, x, y, z] (4通道)
        if points.shape[1] >= 4:
            # 使用完整的4通道（intensity, x, y, z）
            data = points[:, :4].copy()
        elif points.shape[1] == 3:
            # 如果只有3通道（xyz），添加一个零通道作为强度
            intensity = np.zeros((points.shape[0], 1), dtype=points.dtype)
            data = np.concatenate([intensity, points], axis=1)  # (N, 4): [intensity, x, y, z]
        else:
            raise ValueError(f"不支持的点云通道数: {points.shape[1]}，期望3或4通道")

        # 移除无效点（xyz坐标全零或 NaN）
        # 只检查 xyz 坐标（第1-3列），强度值可能为0是正常的
        xyz_coords = data[:, 1:4]  # x, y, z
        valid_mask = np.any(xyz_coords != 0, axis=1) & ~np.any(np.isnan(data), axis=1)
        if valid_mask.sum() == 0:
            # 如果所有点都无效，返回零点云（4通道）
            data = np.zeros((self.num_points, 4), dtype=np.float32)
        else:
            data = data[valid_mask]

        # 归一化（只归一化 xyz 坐标，不归一化强度）
        if self.normalize:
            xyz_coords = data[:, 1:4]  # x, y, z
            centroid = np.mean(xyz_coords, axis=0)
            xyz_normalized = xyz_coords - centroid
            max_dist = np.max(np.linalg.norm(xyz_normalized, axis=1))
            if max_dist > 1e-6:
                xyz_normalized = xyz_normalized / max_dist
            # 合并归一化后的xyz和原始强度
            data = np.concatenate([data[:, 0:1], xyz_normalized], axis=1)  # [intensity, x_norm, y_norm, z_norm]

        # 采样或填充到固定点数（基于xyz坐标进行最远点采样）
        n_points = data.shape[0]
        if n_points >= self.num_points:
            # 最远点采样（基于xyz坐标）
            xyz_coords = data[:, 1:4]
            indices = self._farthest_point_sample(xyz_coords, self.num_points)
            data = data[indices]
        else:
            # 重复采样以填充
            indices = np.random.choice(n_points, self.num_points, replace=True)
            data = data[indices]

        # 分离坐标和特征
        # pos: (N, 3) - xyz坐标
        # feat: (N, 1) - intensity特征
        pos = torch.from_numpy(data[:, 1:4]).float()  # x, y, z
        feat = torch.from_numpy(data[:, 0:1]).float()  # intensity

        result = {"pos": pos, "feat": feat, "path": str(path)}
        return result

    @staticmethod
    def _farthest_point_sample(points: np.ndarray, n_samples: int) -> np.ndarray:
        """最远点采样（简化版）。"""
        if n_samples <= 0 or points.shape[0] <= n_samples:
            return np.arange(points.shape[0])

        centroids = np.zeros((n_samples,), dtype=np.int32)
        distance = np.ones((points.shape[0],), dtype=np.float32) * 1e10
        farthest = np.random.randint(0, points.shape[0])

        for i in range(n_samples):
            centroids[i] = farthest
            centroid = points[farthest, :]
            dist = np.sum((points - centroid) ** 2, axis=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)

        return centroids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 PointNeXt 对点云子集提取特征"
    )
    parser.add_argument(
        "--point-cloud-dir",
        type=str,
        help="点云 .npy 文件所在目录（可以是单个序列的检测结果目录）",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        help="检测元数据 CSV 文件路径（包含 points_path 列）",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="数据集根目录（当使用 metadata-csv 且 points_path 为相对路径时使用）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="特征保存路径（*.pt）",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="PointNeXt/cfgs/scanobjectnn/pointnext-s.yaml",
        help="PointNeXt 模型配置文件路径",
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        help="预训练模型权重路径（可选，如果不提供则使用随机初始化）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="推理批大小",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 的工作线程数",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1024,
        help="每个点云采样的点数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cuda、cuda:0、cpu",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="归一化点云坐标（默认启用）",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="禁用点云归一化",
    )
    return parser.parse_args()


def collect_point_clouds(
    point_cloud_dir: Optional[str] = None,
    metadata_csv: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> Tuple[List[Path], Optional[pd.DataFrame]]:
    """
    收集点云文件路径。

    支持两种方式：
    1. 直接指定点云目录
    2. 通过元数据 CSV 文件（包含 points_path 列）
    """
    if point_cloud_dir:
        point_cloud_dir = Path(point_cloud_dir).expanduser()
        if not point_cloud_dir.exists():
            raise FileNotFoundError(f"未找到点云目录: {point_cloud_dir}")
        point_cloud_paths = sorted(
            [p for p in point_cloud_dir.rglob("*.npy") if p.is_file()]
        )
        if not point_cloud_paths:
            raise RuntimeError(f"目录中未找到 .npy 文件: {point_cloud_dir}")
        return point_cloud_paths, None

    elif metadata_csv:
        metadata_path = Path(metadata_csv).expanduser()
        if not metadata_path.exists():
            raise FileNotFoundError(f"未找到元数据文件: {metadata_path}")
        df = pd.read_csv(metadata_path)

        if "points_path" not in df.columns:
            raise ValueError("CSV 文件必须包含 'points_path' 列")

        if dataset_root:
            dataset_root = Path(dataset_root).expanduser()
            # points_path在CSV中可能是：
            # 1. 完整路径（包含序列名）: seq0002/detections/xxx.npy
            # 2. 相对路径（不包含序列名）: detections/xxx.npy
            # 需要正确构建完整路径: dataset_root/seq0002/detections/xxx.npy
            point_cloud_paths = []
            if "sequence_name" in df.columns:
                for idx, row in df.iterrows():
                    seq_name = row["sequence_name"]
                    rel_path = row["points_path"]
                    
                    # 检查 points_path 是否已经包含序列名
                    if str(rel_path).startswith(seq_name + "/"):
                        # points_path 已经包含序列名，直接使用
                        # 尝试在train和val中查找
                        found = False
                        for split in ["train", "val"]:
                            full_path = dataset_root / split / rel_path
                            if full_path.exists():
                                point_cloud_paths.append(full_path)
                                found = True
                                break
                        if not found:
                            # 如果找不到，尝试直接拼接（dataset_root可能已经是train或val）
                            full_path = dataset_root / rel_path
                            point_cloud_paths.append(full_path)
                else:
                    # points_path 不包含序列名，需要拼接序列名
                    # 尝试在train和val中查找
                    found = False
                    for split in ["train", "val"]:
                        full_path = dataset_root / split / seq_name / rel_path
                        if full_path.exists():
                            point_cloud_paths.append(full_path)
                            found = True
                            break
                    if not found:
                        # 如果找不到，尝试直接拼接
                        full_path = dataset_root / seq_name / rel_path
                        point_cloud_paths.append(full_path)
            else:
                # 如果没有sequence_name，尝试从路径推断
                for path in df["points_path"]:
                    # 尝试在train和val中查找
                    found = False
                    for split in ["train", "val"]:
                        full_path = dataset_root / split / path
                        if full_path.exists():
                            point_cloud_paths.append(full_path)
                            found = True
                            break
                    if not found:
                        point_cloud_paths.append(dataset_root / path)
        else:
            # 假设 points_path 是绝对路径或相对于 CSV 文件所在目录
            csv_dir = metadata_path.parent
            point_cloud_paths = [
                csv_dir / path if not Path(path).is_absolute() else Path(path)
                for path in df["points_path"]
            ]

        # 过滤存在的文件
        valid_paths = [p for p in point_cloud_paths if p.exists()]
        if not valid_paths:
            raise RuntimeError("未找到有效的点云文件")
        if len(valid_paths) < len(point_cloud_paths):
            print(
                f"警告: {len(point_cloud_paths) - len(valid_paths)} 个文件不存在，已跳过"
            )

        return valid_paths, df.iloc[[i for i, p in enumerate(point_cloud_paths) if p.exists()]]

    else:
        raise ValueError("必须提供 --point-cloud-dir 或 --metadata-csv")


def create_model(cfg_path: str, pretrained_path: Optional[str], device: str) -> torch.nn.Module:
    """创建并加载 PointNeXt 模型。"""
    cfg = EasyConfig()
    cfg.load(cfg_path, recursive=True)

    # 构建模型（只使用 encoder，不使用分类头）
    encoder_args = cfg.model.encoder_args
    model = build_model_from_cfg({"NAME": "BaseCls", "encoder_args": encoder_args})

    # 加载预训练权重
    if pretrained_path:
        pretrained_path = Path(pretrained_path).expanduser()
        if not pretrained_path.exists():
            raise FileNotFoundError(f"未找到预训练模型: {pretrained_path}")
        # 只加载 encoder 的权重
        checkpoint = torch.load(pretrained_path, map_location=device)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 过滤出 encoder 的权重
        encoder_state_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        if not encoder_state_dict:
            # 如果没有 encoder. 前缀，尝试直接加载
            encoder_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith("prediction.") and not k.startswith("head.")
            }

        model.encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"已加载预训练权重: {pretrained_path}")

    model.eval()
    model.to(device)
    return model


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    point_cloud_paths: List[Path],
) -> Tuple[torch.Tensor, List[str]]:
    """提取点云特征。"""
    features: List[torch.Tensor] = []
    paths: List[str] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            pos = batch["pos"].to(device)  # (B, N, 3) - xyz坐标
            feat = batch["feat"].to(device)  # (B, N, 1) - intensity特征

            # 模型期望4通道输入特征 (in_channels: 4)
            # 将intensity扩展为4通道：使用xyz坐标作为前3个通道，intensity作为第4个通道
            # 这样可以利用位置信息和强度信息
            xyz_feat = pos  # (B, N, 3) - xyz坐标作为特征
            intensity = feat  # (B, N, 1) - intensity
            feat_4ch = torch.cat([xyz_feat, intensity], dim=2)  # (B, N, 4)
            
            # 将特征转换为 (B, C, N) 格式
            # PointNeXt 期望特征格式为 (B, C, N)
            feat = feat_4ch.transpose(1, 2)  # (B, 4, N)

            # 使用 encoder 提取特征
            # encoder.forward_cls_feat(p0, f0) 其中 p0 是坐标，f0 是特征
            global_feat = model.encoder.forward_cls_feat(pos, feat)
            features.append(global_feat.cpu())

            # 获取对应的路径
            batch_start = batch_idx * dataloader.batch_size
            batch_end = min(batch_start + len(pos), len(point_cloud_paths))
            batch_paths = point_cloud_paths[batch_start:batch_end]
            paths.extend([str(p) for p in batch_paths])

    return torch.cat(features, dim=0), paths


def main():
    args = parse_args()

    # 收集点云文件
    point_cloud_paths, metadata_df = collect_point_clouds(
        point_cloud_dir=args.point_cloud_dir,
        metadata_csv=getattr(args, "metadata_csv", None),
        dataset_root=getattr(args, "dataset_root", None),
    )

    print(f"找到 {len(point_cloud_paths)} 个点云文件")

    # 创建数据集和数据加载器
    dataset = PointCloudDataset(
        point_cloud_paths=point_cloud_paths,
        metadata=metadata_df,
        num_points=args.num_points,
        normalize=args.normalize,
    )

    use_cuda = args.device.startswith("cuda")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=use_cuda,
    )

    # 创建模型
    cfg_path = Path(args.cfg).expanduser()
    if not cfg_path.exists():
        # 尝试相对于项目根目录
        cfg_path = Path(__file__).parent.parent.parent / args.cfg
        if not cfg_path.exists():
            raise FileNotFoundError(f"未找到配置文件: {args.cfg}")

    model = create_model(
        str(cfg_path),
        getattr(args, "pretrained_path", None),
        args.device,
    )

    # 提取特征
    features, paths = extract_features(model, dataloader, args.device, point_cloud_paths)

    # 准备元数据
    metadata = []
    for i, path_str in enumerate(paths):
        path = Path(path_str)
        metadata.append(
            {
                "path": path_str,
                "name": path.name,
                "sequence": path.parent.parent.name if len(path.parts) > 2 else "",
            }
        )

    # 如果提供了元数据 DataFrame，添加额外信息
    if metadata_df is not None:
        for i, meta in enumerate(metadata):
            if i < len(metadata_df):
                row = metadata_df.iloc[i]
                for col in metadata_df.columns:
                    if col not in ["points_path"]:
                        meta[col] = row[col]

    # 保存特征
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "features": features,
            "metadata": metadata,
            "model": "pointnext",
            "cfg": str(cfg_path),
            "num_points": args.num_points,
            "normalize": args.normalize,
        },
        output_path,
    )
    print(f"特征已保存至: {output_path} （共 {features.shape[0]} 个点云）")
    print(f"特征维度: {features.shape[1]}")


if __name__ == "__main__":
    main()

