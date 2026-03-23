#!/usr/bin/env python3
"""
测试缺失模态鲁棒性实验脚本

在验证集上测试不同缺失率下的模型性能
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import json
from collections import defaultdict

from multimodal_window_dataset import MultimodalWindowDataset
from lightweight_classifier import TinyFusion, CompactTransformer, EfficientFusion
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="测试缺失模态鲁棒性")
    parser.add_argument("--model-path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--base-dir", type=str, default="/home/p/MMUAV/data", help="数据集根目录")
    parser.add_argument("--timeline-dir", type=str, default="./out", help="时间线目录")
    parser.add_argument("--features-dir", type=str, default="./features", help="特征文件目录")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="数据集分割")
    parser.add_argument("--missing-rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5], 
                        help="缺失率列表")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--output-dir", type=str, default="./experiments_paper/results/missing_robustness",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    return parser.parse_args()


def randomly_mask_modalities(mask_dict, missing_rate):
    """
    随机屏蔽一定比例的模态
    
    Args:
        mask_dict: 原始模态掩码字典 {modality: mask_tensor}
        missing_rate: 缺失率 (0.0-1.0)
    
    Returns:
        修改后的掩码字典
    """
    masked_dict = {}
    for modality, mask in mask_dict.items():
        if missing_rate == 0.0:
            masked_dict[modality] = mask
        else:
            # 随机选择 missing_rate 比例的样本进行屏蔽
            n_samples = mask.shape[0]
            n_mask = int(n_samples * missing_rate)
            mask_indices = np.random.choice(n_samples, size=n_mask, replace=False)
            
            masked_mask = mask.clone()
            masked_mask[mask_indices] = 0  # 设置为缺失
            
            masked_dict[modality] = masked_mask
    
    return masked_dict


def evaluate_with_missing_rate(model, dataloader, device, missing_rate, num_trials=5):
    """
    在给定缺失率下评估模型性能
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        missing_rate: 缺失率
        num_trials: 重复试验次数（取平均）
    
    Returns:
        平均准确率
    """
    model.eval()
    
    all_accuracies = []
    
    for trial in range(num_trials):
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 获取数据和掩码
                image_feat = batch['image_feat'].to(device)
                livox_feat = batch['livox_feat'].to(device)
                lidar360_feat = batch['lidar360_feat'].to(device)
                radar_feat = batch['radar_feat'].to(device)
                
                image_mask = batch['image_mask'].to(device)
                livox_mask = batch['livox_mask'].to(device)
                lidar360_mask = batch['lidar360_mask'].to(device)
                radar_mask = batch['radar_mask'].to(device)
                
                labels = batch['label'].to(device)
                
                # 应用缺失率
                if missing_rate > 0.0:
                    mask_dict = {
                        'image': image_mask,
                        'livox': livox_mask,
                        'lidar360': lidar360_mask,
                        'radar': radar_mask
                    }
                    masked_dict = randomly_mask_modalities(mask_dict, missing_rate)
                    
                    image_mask = masked_dict['image']
                    livox_mask = masked_dict['livox']
                    lidar360_mask = masked_dict['lidar360']
                    radar_mask = masked_dict['radar']
                
                # 前向传播
                outputs = model(
                    image_feat, livox_feat, lidar360_feat, radar_feat,
                    image_mask, livox_mask, lidar360_mask, radar_mask,
                    batch.get('image_conf', None),
                    batch.get('livox_conf', None),
                    batch.get('lidar360_conf', None),
                    batch.get('radar_conf', None),
                    batch.get('livox_point_count', None),
                    batch.get('lidar360_point_count', None),
                    batch.get('radar_point_count', None),
                )
                
                # 计算准确率
                preds = outputs.argmax(dim=1)
                correct += (preds == labels.argmax(dim=1)).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        all_accuracies.append(accuracy)
    
    # 返回平均准确率
    return np.mean(all_accuracies), np.std(all_accuracies)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    print(f"加载 {args.split} 数据集...")
    dataset = MultimodalWindowDataset(
        timeline_dir=Path(args.timeline_dir),
        base_dir=Path(args.base_dir),
        image_features_path=Path(args.features_dir) / f"image_features_{args.split}.pt",
        livox_features_path=Path(args.features_dir) / f"livox_features_{args.split}.pt",
        lidar360_features_path=Path(args.features_dir) / f"lidar360_features_{args.split}.pt",
        radar_features_path=Path(args.features_dir) / f"radar_features_{args.split}.pt",
        train=(args.split == "train"),
        split_config_path=Path(args.features_dir) / "dataset_split_config.json",
        target_split=args.split,
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        model_type = model_config.get('model_type', 'compact')
    else:
        model_type = 'compact'
    
    # 获取特征维度
    sample = dataset[0]
    image_dim = sample['image_feat'].shape[-1] if 'image_feat' in sample else 768
    livox_dim = sample['livox_feat'].shape[-1] if 'livox_feat' in sample else 512
    lidar360_dim = sample['lidar360_feat'].shape[-1] if 'lidar360_feat' in sample else 512
    radar_dim = sample['radar_feat'].shape[-1] if 'radar_feat' in sample else 512
    
    # 创建模型
    if model_type == 'tiny':
        model = TinyFusion(
            image_dim=image_dim,
            livox_dim=livox_dim,
            lidar360_dim=lidar360_dim,
            radar_dim=radar_dim,
            num_classes=4
        ).to(device)
    elif model_type == 'compact':
        model = CompactTransformer(
            image_dim=image_dim,
            livox_dim=livox_dim,
            lidar360_dim=lidar360_dim,
            radar_dim=radar_dim,
            num_classes=4
        ).to(device)
    elif model_type == 'efficient':
        model = EfficientFusion(
            image_dim=image_dim,
            livox_dim=livox_dim,
            lidar360_dim=lidar360_dim,
            radar_dim=radar_dim,
            num_classes=4
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 加载权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"模型类型: {model_type}")
    print(f"测试缺失率: {args.missing_rates}")
    print()
    
    # 评估不同缺失率
    results = []
    baseline_accuracy = None
    
    for missing_rate in args.missing_rates:
        print(f"测试缺失率: {missing_rate*100:.0f}%...")
        accuracy, std = evaluate_with_missing_rate(model, dataloader, device, missing_rate)
        
        if missing_rate == 0.0:
            baseline_accuracy = accuracy
        
        relative_drop = (baseline_accuracy - accuracy) if baseline_accuracy is not None else 0.0
        
        results.append({
            'missing_rate': missing_rate,
            'accuracy': accuracy,
            'std': std,
            'relative_drop': relative_drop
        })
        
        print(f"  准确率: {accuracy*100:.2f}% ± {std*100:.2f}%")
        if baseline_accuracy is not None:
            print(f"  相对下降: {relative_drop*100:.2f}%")
        print()
    
    # 保存结果
    results_file = output_dir / "missing_robustness_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成表格
    print("="*60)
    print("缺失模态鲁棒性测试结果")
    print("="*60)
    print(f"{'缺失率':<10} {'准确率':<15} {'相对下降':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['missing_rate']*100:>6.0f}%   {r['accuracy']*100:>8.2f}%±{r['std']*100:.2f}%   {r['relative_drop']*100:>10.2f}%")
    
    # 保存表格到文件
    table_file = output_dir / "missing_robustness_table.txt"
    with open(table_file, 'w') as f:
        f.write("缺失模态鲁棒性测试结果\n")
        f.write("="*60 + "\n")
        f.write(f"{'缺失率':<10} {'准确率':<15} {'相对下降':<15}\n")
        f.write("-"*60 + "\n")
        for r in results:
            f.write(f"{r['missing_rate']*100:>6.0f}%   {r['accuracy']*100:>8.2f}%±{r['std']*100:.2f}%   {r['relative_drop']*100:>10.2f}%\n")
    
    print()
    print(f"结果已保存到: {results_file}")
    print(f"表格已保存到: {table_file}")


if __name__ == "__main__":
    main()

