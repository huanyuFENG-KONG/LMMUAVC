"""
评估多模态融合分类器（支持4类分类和软件层面重组）。

该脚本会：
1. 加载训练好的模型
2. 在指定数据集（train/val/test）上进行评估
3. 生成分类报告（精确率、召回率、F1分数）
4. 生成混淆矩阵（归一化和数值）
5. 支持软件层面重组的数据集
"""

import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from multimodal_window_dataset import MultimodalWindowDataset
from multimodal_classifier import create_classifier
from lightweight_classifier import create_lightweight_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="评估多模态融合分类器（4类分类）")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="已训练模型的路径（.pt文件）",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/p/MMUAV/data",
        help="数据集根目录",
    )
    parser.add_argument(
        "--timeline-dir",
        type=str,
        default="./out",
        help="对齐后的时间线CSV文件目录",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="./features",
        help="特征文件目录",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="要评估的数据集分割",
    )
    parser.add_argument(
        "--eval-both",
        action="store_true",
        help="同时评估验证集和测试集并对比结果",
    )
    parser.add_argument(
        "--split-config",
        type=str,
        help="数据集划分配置文件路径（用于软件层面重组）",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=0.4,
        help="时间窗口大小（秒）",
    )
    parser.add_argument(
        "--aggregation-mode",
        type=str,
        default="mean",
        choices=["mean", "max", "attention", "concat"],
        help="特征聚合模式",
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="cross_attention",
        choices=["cross_attention", "concat", "add", "gated"],
        help="模型融合模式",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批处理大小",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="设备 (cuda 或 cpu)",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["Class_0", "Class_1", "Class_2", "Class_3"],
        help="类别名称列表（4个类别）",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="类别数（默认4）",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        choices=["image", "livox", "lidar360", "radar"],
        default=None,
        help="指定使用的模态（默认：所有四种模态）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="评估结果输出目录",
    )
    return parser.parse_args()


def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    use_lightweight: bool = False,
) -> tuple:
    """在数据集上进行评估并返回预测和真实标签。"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"评估中")
        for batch in pbar:
            labels = batch['label'].to(device)
            
            if use_lightweight:
                # 轻量级模型使用四模态接口
                image_feat = batch.get('image_feat', torch.zeros(len(labels), 768)).to(device)
                livox_feat = batch.get('livox_feat', torch.zeros(len(labels), 512)).to(device)
                lidar360_feat = batch.get('lidar360_feat', torch.zeros(len(labels), 512)).to(device)
                radar_feat = batch.get('radar_feat', torch.zeros(len(labels), 512)).to(device)
                
                image_mask = batch.get('image_mask', torch.ones(len(labels), dtype=torch.float32)).to(device)
                livox_mask = batch.get('livox_mask', torch.ones(len(labels), dtype=torch.float32)).to(device)
                lidar360_mask = batch.get('lidar360_mask', torch.ones(len(labels), dtype=torch.float32)).to(device)
                radar_mask = batch.get('radar_mask', torch.ones(len(labels), dtype=torch.float32)).to(device)
                
                # 将置信度和点云数量移动到设备上
                image_conf = batch.get('image_conf', None)
                livox_conf = batch.get('livox_conf', None)
                lidar360_conf = batch.get('lidar360_conf', None)
                radar_conf = batch.get('radar_conf', None)
                livox_point_count = batch.get('livox_point_count', None)
                lidar360_point_count = batch.get('lidar360_point_count', None)
                radar_point_count = batch.get('radar_point_count', None)
                
                if image_conf is not None:
                    image_conf = image_conf.to(device)
                if livox_conf is not None:
                    livox_conf = livox_conf.to(device)
                if lidar360_conf is not None:
                    lidar360_conf = lidar360_conf.to(device)
                if radar_conf is not None:
                    radar_conf = radar_conf.to(device)
                if livox_point_count is not None:
                    livox_point_count = livox_point_count.to(device)
                if lidar360_point_count is not None:
                    lidar360_point_count = lidar360_point_count.to(device)
                if radar_point_count is not None:
                    radar_point_count = radar_point_count.to(device)
                
                logits = model(
                    image_feat, livox_feat, lidar360_feat, radar_feat,
                    image_mask, livox_mask, lidar360_mask, radar_mask,
                    image_conf,
                    livox_conf,
                    lidar360_conf,
                    radar_conf,
                    livox_point_count,
                    lidar360_point_count,
                    radar_point_count,
                )
            else:
                # 旧模型使用双模态接口（向后兼容）
                pointnext_feat = batch.get('pointnext_feat', batch.get('livox_feat', torch.zeros(len(labels), 512))).to(device)
                convnext_feat = batch.get('convnext_feat', batch.get('image_feat', torch.zeros(len(labels), 768))).to(device)
                logits = model(pointnext_feat, convnext_feat)
            
            preds = logits.argmax(dim=1)
            
            # 检查标签格式（应该是one-hot编码）
            if labels.dim() == 2 and labels.shape[1] > 1:
                # one-hot编码，使用argmax
                label_indices = labels.argmax(dim=1)
            elif labels.dim() == 1:
                # 已经是类别索引
                label_indices = labels.long()
            else:
                raise ValueError(f"意外的标签格式: shape={labels.shape}")
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())
            
            # 计算准确率
            if len(all_labels) > 0:
                correct = sum(np.array(all_preds) == np.array(all_labels))
                acc = 100 * correct / len(all_labels)
                pbar.set_postfix({'acc': f'{acc:.2f}%', 'samples': len(all_labels)})
    
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    output_path: Path,
    normalize: bool = False,
):
    """绘制并保存混淆矩阵。"""
    # 检查实际存在的类别
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    
    if len(unique_labels) == 1:
        print(f"⚠️  警告: 数据集中只有1个类别，无法绘制混淆矩阵")
        return
    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    
    if normalize:
        # 避免除以0
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm.astype('float') / row_sums
        title = "归一化混淆矩阵"
        fmt = '.2f'
    else:
        title = "混淆矩阵"
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"混淆矩阵已保存至: {output_path}")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
):
    """打印分类报告。"""
    print("\n" + "=" * 70)
    print("分类报告")
    print("=" * 70)
    
    # 检查实际类别数量
    unique_labels = np.unique(y_true)
    print(f"检测到 {len(unique_labels)} 个类别: {unique_labels}")
    
    if len(unique_labels) != len(class_names):
        print(f"警告: 指定了 {len(class_names)} 个类别名称，但数据集中只有 {len(unique_labels)} 个类别")
    
    if len(unique_labels) == 1:
        print(f"\n⚠️  警告: 数据集中只有1个类别 ({class_names[unique_labels[0]]})")
        print("无法生成混淆矩阵和分类报告（所有样本都是同一类别）")
        print("这可能是数据加载或类别标签的问题")
        return
    
    # 生成分类报告
    report = classification_report(
        y_true,
        y_pred,
        labels=sorted(unique_labels),
        target_names=[class_names[i] for i in sorted(unique_labels)],
        digits=4,
    )
    print(report)


def main():
    args = parse_args()
    
    # 创建评估结果目录
    eval_output_dir = Path(args.output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 自动查找特征文件（支持四模态）
    features_dir = Path(args.features_dir)
    
    # 确定使用的模态
    if args.modalities is not None:
        selected_modalities = set(args.modalities)
        print(f"\n使用指定模态: {', '.join(sorted(selected_modalities))}")
    else:
        selected_modalities = {'image', 'livox', 'lidar360', 'radar'}
    
    # 特征文件路径（四模态）
    image_features_path = features_dir / f"image_features_{args.split}.pt"
    livox_features_path = features_dir / f"livox_features_{args.split}.pt"
    lidar360_features_path = features_dir / f"lidar360_features_{args.split}.pt"
    radar_features_path = features_dir / f"radar_features_{args.split}.pt"
    
    # 向后兼容：尝试旧的命名
    if not image_features_path.exists():
        image_features_path = features_dir / f"convnext_features_{args.split}.pt"
    if not livox_features_path.exists():
        livox_features_path = features_dir / f"pointnext_features_{args.split}.pt"
    
    # 根据选择的模态过滤特征路径
    if 'image' not in selected_modalities:
        image_features_path = None
    if 'livox' not in selected_modalities:
        livox_features_path = None
    if 'lidar360' not in selected_modalities:
        lidar360_features_path = None
    if 'radar' not in selected_modalities:
        radar_features_path = None
    
    # 检查是否至少指定了一个有效模态（如果使用了--modalities参数）
    valid_modalities = {'image', 'livox', 'lidar360', 'radar'}
    if selected_modalities:
        # 如果指定了模态，至少需要指定一个有效模态
        if not selected_modalities.intersection(valid_modalities):
            raise ValueError(f"至少需要指定一个有效模态，可选: {valid_modalities}")
    
    # 创建数据集
    modality_list = ', '.join(sorted(selected_modalities)) if args.modalities else "四模态（图像、Livox、Lidar 360、Radar）"
    print(f"\n创建{args.split}数据集（{modality_list}）...")
    
    # 辅助函数：安全检查路径
    def safe_path(path):
        if path is None:
            return None
        return path if path.exists() else None
    
    dataset = MultimodalWindowDataset(
        timeline_dir=Path(args.timeline_dir),
        base_dir=Path(args.base_dir),
        image_features_path=safe_path(image_features_path),
        livox_features_path=safe_path(livox_features_path),
        lidar360_features_path=safe_path(lidar360_features_path),
        radar_features_path=safe_path(radar_features_path),
        train=(args.split == "train"),
        window_size=args.window_size,
        aggregation_mode=args.aggregation_mode,
        use_precomputed_features=True,
        split_config_path=Path(args.split_config) if args.split_config else None,
        target_split=args.split,
    )
    
    # 打印数据集统计信息
    actual_base_dir = Path(args.base_dir) / args.split if args.split in ['train', 'val', 'test'] else Path(args.base_dir)
    actual_timeline_dir = Path(args.timeline_dir) / args.split if args.split in ['train', 'val', 'test'] else Path(args.timeline_dir)
    
    print(f"\n数据集统计信息:")
    print(f"  数据集大小: {len(dataset)} 个样本")
    print(f"  使用的split: {args.split}")
    print(f"  base_dir: {actual_base_dir} {'✅存在' if actual_base_dir.exists() else '❌不存在'}")
    print(f"  timeline_dir: {actual_timeline_dir} {'✅存在' if actual_timeline_dir.exists() else '❌不存在'}")
    
    # 警告：如果在训练集上评估
    if args.split == 'train':
        print(f"\n⚠️  警告: 您正在训练集上评估模型！")
        print(f"  训练集上的准确率不能代表模型的真实性能。")
        print(f"  请使用 '--split val' 或 '--split test' 来评估模型。")
    
    # 检查前几个样本的标签分布（用于调试）
    print(f"\n检查前10个样本的标签分布...")
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        label = sample['label']
        label_idx = label.argmax().item()
        label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
        if i < 3:  # 只打印前3个样本的详细信息
            print(f"  样本 {i}: label={label_idx}, one-hot={label.cpu().numpy()}, seq={sample.get('seq', 'N/A').item() if hasattr(sample.get('seq', None), 'item') else 'N/A'}")
    print(f"  前10个样本标签分布: {label_counts}")
    
    if len(dataset) == 0:
        raise ValueError(f"数据集为空！请检查数据路径和split配置。split={args.split}, base_dir={args.base_dir}, timeline_dir={args.timeline_dir}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 4,
        pin_memory=True,
    )
    print(f"数据集大小: {len(dataset)}")
    
    # 从数据集获取特征维度（支持四模态）
    sample = dataset[0]
    image_dim = sample.get('image_feat', torch.zeros(768)).shape[0] if 'image' in selected_modalities else 768
    livox_dim = sample.get('livox_feat', torch.zeros(512)).shape[0] if 'livox' in selected_modalities else 512
    lidar360_dim = sample.get('lidar360_feat', torch.zeros(512)).shape[0] if 'lidar360' in selected_modalities else 512
    radar_dim = sample.get('radar_feat', torch.zeros(512)).shape[0] if 'radar' in selected_modalities else 512
    
    print(f"\n特征维度:")
    if 'image' in selected_modalities:
        print(f"  图像（ConvNeXt）: {image_dim}")
    if 'livox' in selected_modalities:
        print(f"  Livox（PointNeXt）: {livox_dim}")
    if 'lidar360' in selected_modalities:
        print(f"  Lidar 360（PointNeXt）: {lidar360_dim}")
    if 'radar' in selected_modalities:
        print(f"  Radar（PointNeXt）: {radar_dim}")
    
    # 向后兼容：为旧模型提供维度
    pointnext_dim = livox_dim  # 默认使用 Livox 维度
    convnext_dim = image_dim
    
    # 加载模型
    print("\n加载模型...")
    model_path = args.model_path
    
    # 如果指定的模型文件不存在，尝试查找最佳模型或最新检查点
    if not os.path.exists(model_path):
        model_dir = os.path.dirname(model_path)
        
        # 尝试多种可能的文件名（.pth 和 .pt）
        possible_best_models = [
            os.path.join(model_dir, 'best_model.pth'),
            os.path.join(model_dir, 'best_model.pt'),
        ]
        
        best_model_path = None
        for path in possible_best_models:
            if os.path.exists(path):
                best_model_path = path
                break
        
        if best_model_path:
            print(f"⚠️  指定的模型文件不存在，使用 {os.path.basename(best_model_path)}: {best_model_path}")
            model_path = best_model_path
        else:
            # 尝试查找最新的检查点文件（支持 .pth 和 .pt）
            import glob
            checkpoint_patterns = [
                os.path.join(model_dir, 'checkpoint_epoch_*.pth'),
                os.path.join(model_dir, 'checkpoint_epoch_*.pt'),
            ]
            checkpoint_files = []
            for pattern in checkpoint_patterns:
                checkpoint_files.extend(glob.glob(pattern))
            
            if checkpoint_files:
                # 按epoch编号排序，取最新的
                checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
                latest_checkpoint = checkpoint_files[-1]
                print(f"⚠️  best_model.pth/.pt 不存在，使用最新检查点: {latest_checkpoint}")
                model_path = latest_checkpoint
            else:
                raise FileNotFoundError(
                    f"模型文件不存在: {args.model_path}\n"
                    f"目录 {model_dir} 中也没有找到 best_model.pth/.pt 或 checkpoint_epoch_*.pth/.pt\n"
                    f"请先训练模型，或者检查模型路径是否正确。\n"
                    f"例如，训练单模态模型可以使用：\n"
                    f"  python train_lightweight.py --modalities {args.modalities[0] if args.modalities and len(args.modalities) > 0 else 'image'} ..."
                )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 检查是否是轻量级模型（通过checkpoint中的字段判断）
    is_lightweight = False
    model_type = None
    
    # 获取 state_dict（用于检查模型结构）
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 方法1: 检查 state_dict 中的键，判断模型类型
    # 旧的两模态模型（MultimodalClassifier）有 fusion_blocks
    # 新的四模态模型（轻量级模型）有 image_missing_emb, livox_missing_emb 等
    has_fusion_blocks = any('fusion_blocks' in key for key in state_dict.keys())
    has_four_modality_embs = any(key in state_dict for key in ['image_missing_emb', 'livox_missing_emb', 'lidar360_missing_emb', 'radar_missing_emb'])
    
    # 方法2: 检查 model_config 或 args 字段
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        # 如果有 model_type 字段，则是轻量级模型
        if 'model_type' in model_config:
            is_lightweight = True
            model_type = model_config['model_type']
        # 如果有 pointnext_dim 和 convnext_dim，则是旧的两模态模型
        elif 'pointnext_dim' in model_config and 'convnext_dim' in model_config:
            is_lightweight = False
        # 否则，根据 state_dict 结构判断
        elif has_four_modality_embs:
            is_lightweight = True
            model_type = 'compact'  # 默认类型
        elif has_fusion_blocks:
            is_lightweight = False
    elif 'args' in checkpoint and 'model_type' in checkpoint['args']:
        is_lightweight = True
        model_type = checkpoint['args']['model_type']
    elif 'model_type' in checkpoint:
        is_lightweight = True
        model_type = checkpoint['model_type']
    elif has_four_modality_embs:
        # 通过 state_dict 结构判断：有四模态嵌入层
        is_lightweight = True
        model_type = 'compact'  # 默认类型
    elif has_fusion_blocks:
        # 通过 state_dict 结构判断：有 fusion_blocks
        is_lightweight = False
    
    if is_lightweight and model_type:
        # 使用轻量级模型（四模态）
        print(f"检测到轻量级模型类型: {model_type}")
        model_kwargs = {}
        
        # 从 model_config 或 args 获取参数
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            if 'hidden_dim' in config and config['hidden_dim'] is not None:
                model_kwargs['hidden_dim'] = int(config['hidden_dim'])
            if 'num_heads' in config and config['num_heads'] is not None:
                model_kwargs['num_heads'] = int(config['num_heads'])
            if 'dropout' in config or 'dropout_rate' in config:
                dropout_val = config.get('dropout') or config.get('dropout_rate')
                if dropout_val is not None:
                    model_kwargs['dropout'] = float(dropout_val)
        elif 'args' in checkpoint:
            args_dict = checkpoint['args']
            if 'hidden_dim' in args_dict and args_dict['hidden_dim'] is not None:
                model_kwargs['hidden_dim'] = int(args_dict['hidden_dim'])
            if 'num_heads' in args_dict and args_dict['num_heads'] is not None:
                model_kwargs['num_heads'] = int(args_dict['num_heads'])
            if 'dropout' in args_dict and args_dict['dropout'] is not None:
                model_kwargs['dropout'] = float(args_dict['dropout'])
        
        if model_kwargs:
            print(f"使用轻量级模型参数: {model_kwargs}")
        else:
            print("使用轻量级模型默认参数")
        
        model = create_lightweight_classifier(
            model_type=model_type,
            image_dim=image_dim,
            livox_dim=livox_dim,
            lidar360_dim=lidar360_dim,
            radar_dim=radar_dim,
            num_classes=args.num_classes,
            **model_kwargs
        ).to(device)
    else:
        # 使用原始大模型（从checkpoint中读取配置，如果没有则使用默认值）
        print("使用原始大模型架构")
        
        # 从checkpoint中读取模型配置
        model_config = checkpoint.get('model_config', {})
        if model_config:
            print(f"从checkpoint读取模型配置: {model_config}")
            pointnext_dim = model_config.get('pointnext_dim', pointnext_dim)
            convnext_dim = model_config.get('convnext_dim', convnext_dim)
            hidden_dim = model_config.get('hidden_dim', 512)
            num_fusion_layers = model_config.get('num_fusion_layers', 2)
            num_heads = model_config.get('num_heads', 8)
            dropout = model_config.get('dropout', 0.1)
            fusion_mode = model_config.get('fusion_mode', args.fusion_mode or 'cross_attention')
        else:
            # 使用默认配置（向后兼容）
            print("⚠️  checkpoint中未找到model_config，使用默认配置")
            hidden_dim = 512
            num_fusion_layers = 2
            num_heads = 8
            dropout = 0.1
            fusion_mode = args.fusion_mode or 'cross_attention'
        
        model = create_classifier(
            pointnext_dim=pointnext_dim,
            convnext_dim=convnext_dim,
            num_classes=args.num_classes,
            hidden_dim=hidden_dim,
            num_fusion_layers=num_fusion_layers,
            num_heads=num_heads,
            dropout=dropout,
            fusion_mode=fusion_mode,
        ).to(device)
    
    # 加载模型权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✅ 模型加载成功")
    
    # 开始评估
    print(f"\n开始评估{args.split}集...")
    all_labels, all_preds = evaluate(model, dataloader, device, use_lightweight=is_lightweight)
    
    # 打印详细统计信息
    print(f"\n评估结果统计:")
    print(f"  总样本数: {len(all_labels)}")
    print(f"  预测的类别: {np.unique(all_preds)} (数量: {[np.sum(all_preds == c) for c in np.unique(all_preds)]})")
    print(f"  真实类别: {np.unique(all_labels)} (数量: {[np.sum(all_labels == c) for c in np.unique(all_labels)]})")
    
    # 检查是否所有标签都是同一个类别（这会导致不合理的准确率）
    unique_labels = np.unique(all_labels)
    if len(unique_labels) == 1:
        print(f"\n⚠️  警告: 数据集中所有样本都属于同一个类别 ({unique_labels[0]})!")
        print(f"  如果模型总是预测这个类别，准确率会显示为100%，但这不代表模型性能好。")
        print(f"  请检查数据集是否正确加载，特别是split参数和数据集路径。")
    
    overall_accuracy = np.mean(all_labels == all_preds) * 100
    print(f"\n总体准确率: {overall_accuracy:.2f}%")
    
    # 如果准确率是100%，额外检查
    if overall_accuracy == 100.0:
        print(f"\n⚠️  警告: 准确率为100%!")
        print(f"  这可能是以下原因之一:")
        print(f"  1. 数据集问题：所有样本属于同一个类别")
        print(f"  2. 数据集加载错误：使用了训练集而不是测试集")
        print(f"  3. 模型过拟合：在训练集上评估")
        print(f"  请检查:")
        print(f"    - split参数是否正确（应该是 'val' 或 'test'，而不是 'train'）")
        print(f"    - 数据集路径是否正确")
        print(f"    - 类别分布是否合理（应该包含多个类别）")
    
    # 打印分类报告
    print_classification_report(all_labels, all_preds, args.class_names)
    
    # 生成混淆矩阵
    print("\n生成混淆矩阵...")
    plot_confusion_matrix(
        all_labels,
        all_preds,
        args.class_names,
        eval_output_dir / f"confusion_matrix_{args.split}.png",
        normalize=False,
    )
    plot_confusion_matrix(
        all_labels,
        all_preds,
        args.class_names,
        eval_output_dir / f"confusion_matrix_normalized_{args.split}.png",
        normalize=True,
    )
    
    # 保存评估结果到文件
    results_file = eval_output_dir / f"evaluation_results_{args.split}.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"多模态分类器评估报告 - {args.split.upper()}集\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write(f"数据集大小: {len(dataset)}\n")
        f.write(f"总体准确率: {overall_accuracy:.2f}%\n\n")
        
        # 检测实际存在的类别
        unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
        num_classes = len(unique_labels)
        
        if num_classes == 1:
            f.write(f"\n警告: 数据集中只有1个类别，无法生成详细的分类报告\n")
        else:
            # 调整类别名称
            if num_classes != len(args.class_names):
                actual_class_names = [args.class_names[i] if i < len(args.class_names) else f"Class_{i}" for i in sorted(unique_labels)]
            else:
                actual_class_names = args.class_names
            
            report = classification_report(
                all_labels,
                all_preds,
                labels=sorted(unique_labels),
                target_names=actual_class_names,
                digits=4,
            )
            f.write(report)
        f.write("\n")
        
        # 混淆矩阵（数值）
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(args.class_names)))
        f.write("\n混淆矩阵 (数值):\n")
        f.write(" " * 15)
        for name in args.class_names:
            f.write(f"{name:>12s}")
        f.write("\n")
        for i, name in enumerate(args.class_names):
            f.write(f"{name:>15s}")
            for j in range(len(args.class_names)):
                f.write(f"{cm[i, j]:>12d}")
            f.write("\n")
        
        # 归一化混淆矩阵
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_normalized = cm.astype('float') / row_sums
        f.write("\n混淆矩阵 (归一化):\n")
        f.write(" " * 15)
        for name in args.class_names:
            f.write(f"{name:>12s}")
        f.write("\n")
        for i, name in enumerate(args.class_names):
            f.write(f"{name:>15s}")
            for j in range(len(args.class_names)):
                f.write(f"{cm_normalized[i, j]:>12.4f}")
            f.write("\n")
    
    print(f"\n评估结果已保存至: {results_file}")
    print(f"混淆矩阵已保存至: {eval_output_dir}")
    print("\n" + "=" * 70)
    print("评估完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

