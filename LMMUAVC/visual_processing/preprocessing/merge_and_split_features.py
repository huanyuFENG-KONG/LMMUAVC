"""
合并序列特征并按类别平衡划分数据集。

该脚本会：
1. 合并所有序列的特征文件
2. 按类别平衡样本数量
3. 确保训练、验证、测试集使用时空分布特征相同的目标（每个split都包含所有类别和序列的样本）
4. 按时间顺序划分，同时保持类别和序列的平衡
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pandas as pd


def load_class_labels(base_dir: Path, seq: int, timestamp: float, tolerance: float = 0.1) -> Optional[int]:
    """
    从类别文件中加载类别标签。
    
    Args:
        base_dir: 数据集根目录
        seq: 序列号
        timestamp: 时间戳
        tolerance: 时间戳容差（秒）
    
    Returns:
        类别ID（0-3）或None
    """
    seq_dir = base_dir / f"seq{seq:04d}"
    class_dir = seq_dir / "class"
    
    if not class_dir.exists():
        return None
    
    # 精确匹配
    class_file = class_dir / f"{timestamp:.7f}.npy"
    if class_file.exists():
        try:
            class_id = int(np.load(class_file))
            return class_id
        except Exception:
            pass
    
    # 容差匹配
    best_match = None
    best_diff = float('inf')
    
    for class_file in class_dir.glob("*.npy"):
        try:
            file_ts = float(class_file.stem)
            diff = abs(file_ts - timestamp)
            if diff < best_diff and diff <= tolerance:
                best_diff = diff
                best_match = class_file
        except ValueError:
            continue
    
    if best_match and best_match.exists():
        try:
            class_id = int(np.load(best_match))
            return class_id
        except Exception:
            pass
    
    return None


def extract_sequence_from_path(file_path: Path) -> Optional[int]:
    """从文件路径中提取序列号。"""
    stem = file_path.stem
    # 查找 seqXXXX 格式
    if 'seq' in stem:
        parts = stem.split('_')
        for part in parts:
            if part.startswith('seq'):
                try:
                    seq_str = part.replace('seq', '')
                    return int(seq_str)
                except ValueError:
                    continue
    return None


def load_all_sequence_features(
    features_dir: Path,
    feature_type: str,  # 'image', 'livox', 'lidar360', 'radar'
    base_dir: Path
) -> Tuple[torch.Tensor, List[Dict], Dict[int, List[int]]]:
    """
    加载所有序列的特征文件并合并。
    
    Returns:
        features: 合并后的特征张量 (N, D)
        metadata: 合并后的元数据列表
        seq_indices: 序列索引映射 {seq_id: [indices]}
    """
    # 查找所有序列的特征文件
    pattern = f"{feature_type}_features_aligned_seq*.pt"
    feature_files = sorted(features_dir.glob(pattern))
    
    if not feature_files:
        # 尝试其他可能的文件名模式
        pattern = f"*{feature_type}*seq*.pt"
        feature_files = sorted(features_dir.glob(pattern))
    
    if not feature_files:
        raise FileNotFoundError(f"未找到 {feature_type} 特征文件: {features_dir / pattern}")
    
    print(f"\n找到 {len(feature_files)} 个 {feature_type} 特征文件")
    
    all_features = []
    all_metadata = []
    seq_indices = defaultdict(list)
    current_idx = 0
    
    for file_path in feature_files:
        # 提取序列号
        seq = extract_sequence_from_path(file_path)
        if seq is None:
            print(f"警告: 无法从路径提取序列号: {file_path}")
            continue
        
        print(f"  加载序列 {seq:04d}: {file_path.name}")
        data = torch.load(file_path, map_location='cpu')
        
        if isinstance(data, dict):
            features = data['features']  # (N, D)
            metadata_list = data.get('metadata', [])
        else:
            features = data
            metadata_list = []
        
        # 为元数据添加序列信息
        for meta in metadata_list:
            if not isinstance(meta, dict):
                continue
            meta['seq'] = seq
        
        # 加载类别标签
        for i, meta in enumerate(metadata_list):
            if not isinstance(meta, dict):
                continue
            timestamp = meta.get('window_center', meta.get('timestamp', 0))
            # 确保timestamp是数值类型
            if isinstance(timestamp, (np.floating, np.integer)):
                timestamp = float(timestamp)
            elif not isinstance(timestamp, (int, float)):
                timestamp = 0.0
            
            if timestamp > 0:
                class_id = load_class_labels(base_dir, seq, timestamp)
                if class_id is not None:
                    meta['class_id'] = int(class_id)
        
        num_samples = features.shape[0]
        all_features.append(features)
        all_metadata.extend(metadata_list)
        
        # 记录序列索引
        seq_indices[seq] = list(range(current_idx, current_idx + num_samples))
        current_idx += num_samples
    
    # 合并特征
    if all_features:
        merged_features = torch.cat(all_features, dim=0)
    else:
        merged_features = torch.empty((0, 512))  # 默认特征维度
    
    print(f"  总样本数: {len(merged_features)}")
    print(f"  序列数: {len(seq_indices)}")
    
    return merged_features, all_metadata, dict(seq_indices)


def balance_split_by_class_and_sequence(
    metadata: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_by_time: bool = True
) -> Tuple[List[int], List[int], List[int], Dict]:
    """
    按类别和序列平衡划分数据集。
    
    策略：
    1. 按类别和序列分组
    2. 对于每个(类别, 序列)组合，按时间顺序划分
    3. 确保每个split都包含所有类别和序列的样本
    
    Returns:
        train_indices, val_indices, test_indices, statistics
    """
    # 验证比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为1.0，当前为 {total_ratio}")
    
    # 按类别和序列分组
    grouped_samples = defaultdict(list)
    
    for idx, meta in enumerate(metadata):
        class_id = meta.get('class_id')
        seq = meta.get('seq')
        timestamp = meta.get('window_center', meta.get('timestamp', 0))
        
        if class_id is None or seq is None:
            continue
        
        grouped_samples[(class_id, seq)].append((idx, timestamp))
    
    print(f"\n数据集统计:")
    print(f"  有效样本数: {sum(len(samples) for samples in grouped_samples.values())}")
    print(f"  类别×序列组合数: {len(grouped_samples)}")
    
    # 统计每个类别的样本数
    class_counts = defaultdict(int)
    seq_counts = defaultdict(int)
    for (class_id, seq), samples in grouped_samples.items():
        class_counts[class_id] += len(samples)
        seq_counts[seq] += len(samples)
    
    print(f"\n类别分布:")
    for class_id in sorted(class_counts.keys()):
        print(f"  类别 {class_id}: {class_counts[class_id]} 个样本")
    
    print(f"\n序列分布:")
    for seq in sorted(seq_counts.keys()):
        print(f"  序列 {seq:04d}: {seq_counts[seq]} 个样本")
    
    # 按时间排序并划分
    train_indices = []
    val_indices = []
    test_indices = []
    
    # 记录每个split的统计信息
    split_stats = {
        'train': defaultdict(int),
        'val': defaultdict(int),
        'test': defaultdict(int)
    }
    
    for (class_id, seq), samples in grouped_samples.items():
        # 按时间戳排序
        if split_by_time:
            samples = sorted(samples, key=lambda x: x[1])
        
        n_samples = len(samples)
        n_train = max(1, int(n_samples * train_ratio))
        n_val = max(0, int(n_samples * val_ratio))
        n_test = n_samples - n_train - n_val
        
        # 按比例划分
        indices = [idx for idx, _ in samples]
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
        
        # 更新统计
        split_stats['train'][(class_id, seq)] = len(train_idx)
        split_stats['val'][(class_id, seq)] = len(val_idx)
        split_stats['test'][(class_id, seq)] = len(test_idx)
    
    # 按原始索引排序（保持特征顺序）
    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)
    test_indices = sorted(test_indices)
    
    # 统计每个split的类别分布
    statistics = {
        'train': {
            'total': len(train_indices),
            'class_distribution': defaultdict(int),
            'seq_distribution': defaultdict(int)
        },
        'val': {
            'total': len(val_indices),
            'class_distribution': defaultdict(int),
            'seq_distribution': defaultdict(int)
        },
        'test': {
            'total': len(test_indices),
            'class_distribution': defaultdict(int),
            'seq_distribution': defaultdict(int)
        }
    }
    
    for idx in train_indices:
        meta = metadata[idx]
        class_id = meta.get('class_id')
        seq = meta.get('seq')
        if class_id is not None:
            statistics['train']['class_distribution'][class_id] += 1
        if seq is not None:
            statistics['train']['seq_distribution'][seq] += 1
    
    for idx in val_indices:
        meta = metadata[idx]
        class_id = meta.get('class_id')
        seq = meta.get('seq')
        if class_id is not None:
            statistics['val']['class_distribution'][class_id] += 1
        if seq is not None:
            statistics['val']['seq_distribution'][seq] += 1
    
    for idx in test_indices:
        meta = metadata[idx]
        class_id = meta.get('class_id')
        seq = meta.get('seq')
        if class_id is not None:
            statistics['test']['class_distribution'][class_id] += 1
        if seq is not None:
            statistics['test']['seq_distribution'][seq] += 1
    
    # 转换为普通字典
    for split in ['train', 'val', 'test']:
        statistics[split]['class_distribution'] = dict(statistics[split]['class_distribution'])
        statistics[split]['seq_distribution'] = dict(statistics[split]['seq_distribution'])
    
    return train_indices, val_indices, test_indices, statistics


def save_split_features(
    features: torch.Tensor,
    metadata: List[Dict],
    indices: List[int],
    output_path: Path,
    feature_type: str
):
    """保存划分后的特征文件。"""
    split_features = features[indices]
    split_metadata = [metadata[i] for i in indices]
    
    output_data = {
        'features': split_features,
        'metadata': split_metadata,
        'model': feature_type
    }
    
    torch.save(output_data, output_path)
    print(f"  保存 {feature_type} {output_path.name}: {split_features.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="合并序列特征并按类别平衡划分数据集"
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="./features",
        help="特征文件目录",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/p/MMUAV/data",
        help="数据集根目录（用于加载类别标签）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./features",
        help="输出目录",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="训练集比例（默认0.7）",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="验证集比例（默认0.15）",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="测试集比例（默认0.15）",
    )
    parser.add_argument(
        "--split-by-time",
        action="store_true",
        default=True,
        help="按时间顺序划分（默认启用）",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=["image", "livox", "lidar360", "radar"],
        help="要处理的模态列表（默认: image livox lidar360 radar）",
    )
    
    args = parser.parse_args()
    
    features_dir = Path(args.features_dir)
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("合并序列特征并按类别平衡划分数据集")
    print("=" * 80)
    print(f"特征目录: {features_dir}")
    print(f"数据集根目录: {base_dir}")
    print(f"输出目录: {output_dir}")
    print(f"划分比例: 训练集 {args.train_ratio}, 验证集 {args.val_ratio}, 测试集 {args.test_ratio}")
    print(f"模态: {args.modalities}")
    print()
    
    # 加载所有模态的特征（使用第一个模态来确定划分）
    all_features = {}
    all_metadata = {}
    
    # 使用第一个模态来确定划分索引
    # 注意：由于 align_features_multimodal 已经确保了四种模态的索引对齐
    # （同一索引对应同一时间窗口，使用相同的 metadata），
    # 所以可以使用参考模态的索引来划分所有模态
    reference_modality = args.modalities[0]
    
    print(f"\n加载参考模态特征: {reference_modality}")
    ref_features, ref_metadata, ref_seq_indices = load_all_sequence_features(
        features_dir, reference_modality, base_dir
    )
    
    # 验证参考模态的样本数（用于后续验证其他模态是否对齐）
    ref_num_samples = len(ref_features)
    print(f"  参考模态 ({reference_modality}) 总样本数: {ref_num_samples}")
    
    # 加载其他模态的特征
    for modality in args.modalities:
        if modality == reference_modality:
            all_features[modality] = ref_features
            all_metadata[modality] = ref_metadata
            continue
        
        print(f"\n加载 {modality} 特征")
        try:
            features, metadata, seq_indices = load_all_sequence_features(
                features_dir, modality, base_dir
            )
            # 验证样本数是否对齐（由于 align_features_multimodal 已经对齐，应该相同）
            num_samples = len(features)
            if num_samples != ref_num_samples:
                print(f"  ⚠️  警告: {modality} 模态样本数 ({num_samples}) 与参考模态 ({ref_num_samples}) 不一致")
                print(f"     这可能导致索引不对齐。请检查特征对齐是否正确。")
            else:
                print(f"  ✓ {modality} 模态样本数已对齐: {num_samples}")
            
            all_features[modality] = features
            all_metadata[modality] = metadata
        except FileNotFoundError as e:
            print(f"  警告: {e}，跳过该模态")
            continue
    
    # 使用参考模态的元数据来确定划分
    print(f"\n使用 {reference_modality} 模态的元数据确定数据集划分...")
    train_indices, val_indices, test_indices, statistics = balance_split_by_class_and_sequence(
        ref_metadata,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_by_time=args.split_by_time
    )
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("数据集划分统计")
    print("=" * 80)
    
    for split_name in ['train', 'val', 'test']:
        stats = statistics[split_name]
        print(f"\n{split_name.upper()} 集:")
        print(f"  总样本数: {stats['total']}")
        print(f"  类别分布: {dict(stats['class_distribution'])}")
        print(f"  序列分布: {dict(stats['seq_distribution'])}")
    
    # 保存划分后的特征文件
    print("\n" + "=" * 80)
    print("保存划分后的特征文件")
    print("=" * 80)
    
    for modality in args.modalities:
        if modality not in all_features:
            continue
        
        features = all_features[modality]
        metadata = all_metadata[modality]
        
        # 确保索引在有效范围内
        max_idx = len(features) - 1
        train_idx_filtered = [i for i in train_indices if i <= max_idx]
        val_idx_filtered = [i for i in val_indices if i <= max_idx]
        test_idx_filtered = [i for i in test_indices if i <= max_idx]
        
        print(f"\n{modality} 特征:")
        
        # 保存训练集
        train_path = output_dir / f"{modality}_features_train.pt"
        save_split_features(features, metadata, train_idx_filtered, train_path, modality)
        
        # 保存验证集
        val_path = output_dir / f"{modality}_features_val.pt"
        save_split_features(features, metadata, val_idx_filtered, val_path, modality)
        
        # 保存测试集
        test_path = output_dir / f"{modality}_features_test.pt"
        save_split_features(features, metadata, test_idx_filtered, test_path, modality)
    
    # 保存划分配置（用于数据集类加载）
    split_config = {
        'reference_modality': reference_modality,
        'split_indices': {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        },
        'statistics': statistics
    }
    
    config_path = output_dir / "dataset_split_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(split_config, f, indent=2, ensure_ascii=False)
    print(f"\n划分配置已保存: {config_path}")
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    for modality in args.modalities:
        print(f"  {modality}_features_train.pt")
        print(f"  {modality}_features_val.pt")
        print(f"  {modality}_features_test.pt")
    print(f"  dataset_split_config.json")


if __name__ == "__main__":
    main()

