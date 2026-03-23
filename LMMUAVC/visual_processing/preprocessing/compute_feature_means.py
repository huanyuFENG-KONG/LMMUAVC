"""
计算训练集特征均值（用于 Mean Imputation）

此脚本计算训练集上每个模态的特征均值，用于 Mean Imputation 基线方法。
"""

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from multimodal_window_dataset import MultimodalWindowDataset


def compute_feature_means(
    train_dataset: MultimodalWindowDataset,
    device: str = 'cuda',
    batch_size: int = 64
) -> dict:
    """
    计算训练集上每个模态的特征均值
    
    Args:
        train_dataset: 训练集数据集
        device: 计算设备
        batch_size: 批次大小
    
    Returns:
        dict: 包含每个模态均值的字典
        {
            'image_mean': torch.Tensor,  # [image_dim]
            'livox_mean': torch.Tensor,  # [livox_dim]
            'lidar360_mean': torch.Tensor,  # [lidar360_dim]
            'radar_mean': torch.Tensor,  # [radar_dim]
        }
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 累积特征和计数
    image_sum = None
    livox_sum = None
    lidar360_sum = None
    radar_sum = None
    
    image_count = 0
    livox_count = 0
    lidar360_count = 0
    radar_count = 0
    
    print("计算特征均值...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx % 100 == 0:
            print(f"  处理批次 {batch_idx}/{len(train_loader)}")
        
        # 提取特征
        image_feat = batch['image_feat'].to(device) if 'image_feat' in batch else None
        livox_feat = batch['livox_feat'].to(device) if 'livox_feat' in batch else None
        lidar360_feat = batch['lidar360_feat'].to(device) if 'lidar360_feat' in batch else None
        radar_feat = batch['radar_feat'].to(device) if 'radar_feat' in batch else None
        
        # 提取mask（如果存在）
        image_mask = batch.get('image_mask', None)
        livox_mask = batch.get('livox_mask', None)
        lidar360_mask = batch.get('lidar360_mask', None)
        radar_mask = batch.get('radar_mask', None)
        
        # 累积图像特征（只计算非缺失的）
        if image_feat is not None:
            if image_mask is not None:
                image_mask = image_mask.to(device)
                valid_image = image_mask.sum()
                if valid_image > 0:
                    valid_image_feat = image_feat[image_mask.bool()]
                    if image_sum is None:
                        image_sum = valid_image_feat.sum(dim=0)
                    else:
                        image_sum += valid_image_feat.sum(dim=0)
                    image_count += valid_image.item()
            else:
                # 没有mask，假设所有特征都有效
                if image_sum is None:
                    image_sum = image_feat.sum(dim=0)
                    image_count = image_feat.shape[0]
                else:
                    image_sum += image_feat.sum(dim=0)
                    image_count += image_feat.shape[0]
        
        # 累积Livox特征
        if livox_feat is not None:
            if livox_mask is not None:
                livox_mask = livox_mask.to(device)
                valid_livox = livox_mask.sum()
                if valid_livox > 0:
                    valid_livox_feat = livox_feat[livox_mask.bool()]
                    if livox_sum is None:
                        livox_sum = valid_livox_feat.sum(dim=0)
                    else:
                        livox_sum += valid_livox_feat.sum(dim=0)
                    livox_count += valid_livox.item()
            else:
                if livox_sum is None:
                    livox_sum = livox_feat.sum(dim=0)
                    livox_count = livox_feat.shape[0]
                else:
                    livox_sum += livox_feat.sum(dim=0)
                    livox_count += livox_feat.shape[0]
        
        # 累积Lidar 360特征
        if lidar360_feat is not None:
            if lidar360_mask is not None:
                lidar360_mask = lidar360_mask.to(device)
                valid_lidar360 = lidar360_mask.sum()
                if valid_lidar360 > 0:
                    valid_lidar360_feat = lidar360_feat[lidar360_mask.bool()]
                    if lidar360_sum is None:
                        lidar360_sum = valid_lidar360_feat.sum(dim=0)
                    else:
                        lidar360_sum += valid_lidar360_feat.sum(dim=0)
                    lidar360_count += valid_lidar360.item()
            else:
                if lidar360_sum is None:
                    lidar360_sum = lidar360_feat.sum(dim=0)
                    lidar360_count = lidar360_feat.shape[0]
                else:
                    lidar360_sum += lidar360_feat.sum(dim=0)
                    lidar360_count += lidar360_feat.shape[0]
        
        # 累积Radar特征
        if radar_feat is not None:
            if radar_mask is not None:
                radar_mask = radar_mask.to(device)
                valid_radar = radar_mask.sum()
                if valid_radar > 0:
                    valid_radar_feat = radar_feat[radar_mask.bool()]
                    if radar_sum is None:
                        radar_sum = valid_radar_feat.sum(dim=0)
                    else:
                        radar_sum += valid_radar_feat.sum(dim=0)
                    radar_count += valid_radar.item()
            else:
                if radar_sum is None:
                    radar_sum = radar_feat.sum(dim=0)
                    radar_count = radar_feat.shape[0]
                else:
                    radar_sum += radar_feat.sum(dim=0)
                    radar_count += radar_feat.shape[0]
    
    # 计算均值
    means = {}
    if image_sum is not None and image_count > 0:
        means['image_mean'] = (image_sum / image_count).cpu()
        print(f"图像特征均值: shape={means['image_mean'].shape}, count={image_count}")
    if livox_sum is not None and livox_count > 0:
        means['livox_mean'] = (livox_sum / livox_count).cpu()
        print(f"Livox特征均值: shape={means['livox_mean'].shape}, count={livox_count}")
    if lidar360_sum is not None and lidar360_count > 0:
        means['lidar360_mean'] = (lidar360_sum / lidar360_count).cpu()
        print(f"Lidar 360特征均值: shape={means['lidar360_mean'].shape}, count={lidar360_count}")
    if radar_sum is not None and radar_count > 0:
        means['radar_mean'] = (radar_sum / radar_count).cpu()
        print(f"Radar特征均值: shape={means['radar_mean'].shape}, count={radar_count}")
    
    return means


def main():
    parser = argparse.ArgumentParser(description="计算训练集特征均值（用于 Mean Imputation）")
    parser.add_argument("--features-dir", type=str, required=True,
                        help="特征文件目录")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="数据集根目录")
    parser.add_argument("--timeline-dir", type=str, required=True,
                        help="时间线文件目录")
    parser.add_argument("--split-config", type=str, default=None,
                        help="划分配置文件路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出均值文件路径（.pt格式）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    
    args = parser.parse_args()
    
    # 构建特征文件路径
    features_dir = Path(args.features_dir)
    image_features_path = features_dir / "image_features_train.pt"
    livox_features_path = features_dir / "livox_features_train.pt"
    lidar360_features_path = features_dir / "lidar360_features_train.pt"
    radar_features_path = features_dir / "radar_features_train.pt"
    
    # 创建训练集
    split_config_path = Path(args.split_config) if args.split_config else None
    if split_config_path is None:
        potential_config = features_dir / "dataset_split_config.json"
        if potential_config.exists():
            split_config_path = potential_config
    
    train_dataset = MultimodalWindowDataset(
        timeline_dir=Path(args.timeline_dir),
        base_dir=Path(args.base_dir),
        image_features_path=image_features_path if image_features_path.exists() else None,
        livox_features_path=livox_features_path if livox_features_path.exists() else None,
        lidar360_features_path=lidar360_features_path if lidar360_features_path.exists() else None,
        radar_features_path=radar_features_path if radar_features_path.exists() else None,
        train=True,
        use_precomputed_features=True,
        split_config_path=split_config_path,
        target_split='train'
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    
    # 计算均值
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    means = compute_feature_means(train_dataset, device=device, batch_size=args.batch_size)
    
    # 保存均值
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(means, output_path)
    print(f"\n均值已保存到: {output_path}")
    
    # 打印统计信息
    print("\n特征均值统计:")
    for key, value in means.items():
        print(f"  {key}: shape={value.shape}, mean={value.mean().item():.6f}, std={value.std().item():.6f}")


if __name__ == "__main__":
    main()
