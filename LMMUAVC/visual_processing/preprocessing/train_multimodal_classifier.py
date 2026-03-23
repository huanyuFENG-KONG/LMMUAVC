"""
训练多模态融合分类器。

完整流程：
1. 对齐图像与点云数据（使用preprocessing_improved.py）
2. 提取0.4秒窗口内的点云特征和图像特征
3. 拼接特征
4. 使用Transformer-based分类器进行分类
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from multimodal_window_dataset import MultimodalWindowDataset
from multimodal_classifier import MultimodalClassifier, create_classifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练多模态融合分类器"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/home/dlut-ug/Anti_UAV_data",
        help="数据集根目录",
    )
    parser.add_argument(
        "--timeline-dir",
        type=str,
        default="./out",
        help="对齐后的时间线CSV文件目录",
    )
    parser.add_argument(
        "--pointnext-features",
        type=str,
        help="预提取的PointNeXt特征文件路径（*.pt，训练集）。如果未指定，将自动查找 pointnext_features_train.pt",
    )
    parser.add_argument(
        "--convnext-features",
        type=str,
        help="预提取的ConvNeXt特征文件路径（*.pt，训练集）。如果未指定，将自动查找 convnext_features_train.pt",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="./features",
        help="特征文件目录（用于自动查找特征文件）",
    )
    parser.add_argument(
        "--split-config",
        type=str,
        help="数据集划分配置文件路径（用于软件层面重组）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="模型保存目录",
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
        help="特征融合模式",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批大小",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="训练轮数",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="权重衰减",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader工作线程数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="恢复训练的检查点路径",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="TensorBoard日志目录",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="隐藏层维度",
    )
    parser.add_argument(
        "--num-fusion-layers",
        type=int,
        default=2,
        help="融合层数量",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Transformer注意力头数",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout比率",
    )
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
) -> dict:
    """训练一个epoch。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        pointnext_feat = batch['pointnext_feat'].to(device)
        convnext_feat = batch['convnext_feat'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(pointnext_feat, convnext_feat)
        loss = criterion(logits, labels.argmax(dim=1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        target = labels.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%',
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100 * correct / total,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
) -> dict:
    """验证。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            pointnext_feat = batch['pointnext_feat'].to(device)
            convnext_feat = batch['convnext_feat'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            logits = model(pointnext_feat, convnext_feat)
            loss = criterion(logits, labels.argmax(dim=1))
            
            # 统计
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            target = labels.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%',
            })
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100 * correct / total,
    }


def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 自动查找特征文件（如果未指定）
    features_dir = Path(args.features_dir)
    
    pointnext_features_path = None
    if args.pointnext_features:
        pointnext_features_path = Path(args.pointnext_features)
    else:
        # 优先尝试新的命名（livox_features），然后尝试旧的命名（pointnext_features）
        auto_train_paths = [
            features_dir / "livox_features_train.pt",  # 新命名
            features_dir / "pointnext_features_train.pt",  # 旧命名（向后兼容）
        ]
        for path in auto_train_paths:
            if path.exists():
                pointnext_features_path = path
                print(f"自动找到训练集点云特征: {pointnext_features_path}")
                break
        if pointnext_features_path is None:
            print(f"警告: 未找到点云特征文件（已尝试: {[str(p) for p in auto_train_paths]}），请使用 --pointnext-features 指定")
    
    convnext_features_path = None
    if args.convnext_features:
        convnext_features_path = Path(args.convnext_features)
    else:
        # 优先尝试新的命名（image_features），然后尝试旧的命名（convnext_features）
        auto_train_paths = [
            features_dir / "image_features_train.pt",  # 新命名
            features_dir / "convnext_features_train.pt",  # 旧命名（向后兼容）
        ]
        for path in auto_train_paths:
            if path.exists():
                convnext_features_path = path
                print(f"自动找到训练集图像特征: {convnext_features_path}")
                break
        if convnext_features_path is None:
            print(f"警告: 未找到图像特征文件（已尝试: {[str(p) for p in auto_train_paths]}），请使用 --convnext-features 指定")
    
    # 创建数据集
    print("创建训练集...")
    train_dataset = MultimodalWindowDataset(
        timeline_dir=Path(args.timeline_dir),
        base_dir=Path(args.base_dir),
        pointnext_features_path=pointnext_features_path,
        convnext_features_path=convnext_features_path,
        train=True,
        window_size=args.window_size,
        aggregation_mode=args.aggregation_mode,
        use_precomputed_features=True,
        split_config_path=Path(args.split_config) if args.split_config else None,
        target_split='train',
    )
    
    # 验证集特征文件（可选，如果不存在则使用训练集特征）
    val_pointnext_features_path = None
    val_convnext_features_path = None
    
    # 优先尝试新的命名，然后尝试旧的命名
    val_pointnext_paths = [
        features_dir / "livox_features_val.pt",  # 新命名
        features_dir / "pointnext_features_val.pt",  # 旧命名（向后兼容）
    ]
    for path in val_pointnext_paths:
        if path.exists():
            val_pointnext_features_path = path
            print(f"找到验证集点云特征: {val_pointnext_features_path}")
            break
    
    if val_pointnext_features_path is None:
        print(f"未找到验证集点云特征，将使用训练集特征: {pointnext_features_path}")
        val_pointnext_features_path = pointnext_features_path
    
    val_convnext_paths = [
        features_dir / "image_features_val.pt",  # 新命名
        features_dir / "convnext_features_val.pt",  # 旧命名（向后兼容）
    ]
    for path in val_convnext_paths:
        if path.exists():
            val_convnext_features_path = path
            print(f"找到验证集图像特征: {val_convnext_features_path}")
            break
    
    if val_convnext_features_path is None:
        print(f"未找到验证集图像特征，将使用训练集特征: {convnext_features_path}")
        val_convnext_features_path = convnext_features_path
    
    print("创建验证集...")
    val_dataset = MultimodalWindowDataset(
        timeline_dir=Path(args.timeline_dir),
        base_dir=Path(args.base_dir),
        pointnext_features_path=val_pointnext_features_path,
        convnext_features_path=val_convnext_features_path,
        train=False,
        window_size=args.window_size,
        aggregation_mode=args.aggregation_mode,
        use_precomputed_features=True,
        split_config_path=Path(args.split_config) if args.split_config else None,
        target_split='val',
    )
    
    # 测试集特征文件（可选）
    test_pointnext_features_path = None
    test_convnext_features_path = None
    
    # 优先尝试新的命名，然后尝试旧的命名
    test_pointnext_paths = [
        features_dir / "livox_features_test.pt",  # 新命名
        features_dir / "pointnext_features_test.pt",  # 旧命名（向后兼容）
    ]
    for path in test_pointnext_paths:
        if path.exists():
            test_pointnext_features_path = path
            print(f"找到测试集点云特征: {test_pointnext_features_path}")
            break
    
    test_convnext_paths = [
        features_dir / "image_features_test.pt",  # 新命名
        features_dir / "convnext_features_test.pt",  # 旧命名（向后兼容）
    ]
    for path in test_convnext_paths:
        if path.exists():
            test_convnext_features_path = path
            print(f"找到测试集图像特征: {test_convnext_features_path}")
            break
    
    # 创建测试集数据集（如果特征文件存在）
    test_dataset = None
    if test_pointnext_features_path and test_convnext_features_path:
        print("创建测试集...")
        test_dataset = MultimodalWindowDataset(
            timeline_dir=Path(args.timeline_dir),
            base_dir=Path(args.base_dir),
            pointnext_features_path=test_pointnext_features_path,
            convnext_features_path=test_convnext_features_path,
            train=False,
            window_size=args.window_size,
            aggregation_mode=args.aggregation_mode,
            use_precomputed_features=True,
            split_config_path=Path(args.split_config) if args.split_config else None,
            target_split='test',
        )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    if test_dataset is not None:
        print(f"测试集大小: {len(test_dataset)}")
    else:
        print(f"测试集: 未找到测试集特征文件，跳过")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False,
        )
    
    # 获取特征维度
    # 优先从特征文件中获取维度（更可靠），如果不可用则从数据集样本获取
    pointnext_dim = None
    convnext_dim = None
    
    # 从特征文件中获取维度（如果特征文件存在）
    if pointnext_features_path and Path(pointnext_features_path).exists():
        try:
            pointnext_data = torch.load(pointnext_features_path, map_location='cpu')
            if 'features' in pointnext_data:
                pointnext_dim = pointnext_data['features'].shape[1]
                print(f"从特征文件获取 PointNeXt 维度: {pointnext_dim}")
        except Exception as e:
            print(f"警告: 无法从特征文件获取 PointNeXt 维度: {e}")
    
    if convnext_features_path and Path(convnext_features_path).exists():
        try:
            convnext_data = torch.load(convnext_features_path, map_location='cpu')
            if 'features' in convnext_data:
                convnext_dim = convnext_data['features'].shape[1]
                print(f"从特征文件获取 ConvNeXt 维度: {convnext_dim}")
        except Exception as e:
            print(f"警告: 无法从特征文件获取 ConvNeXt 维度: {e}")
    
    # 如果从特征文件获取失败，从数据集样本获取
    if pointnext_dim is None or convnext_dim is None:
        sample = train_dataset[0]
        if pointnext_dim is None:
            pointnext_dim = sample['pointnext_feat'].shape[0]
            print(f"从数据集样本获取 PointNeXt 维度: {pointnext_dim}")
        if convnext_dim is None:
            convnext_dim = sample['convnext_feat'].shape[0]
            print(f"从数据集样本获取 ConvNeXt 维度: {convnext_dim}")
    
    # 验证维度（ConvNeXt应该是768，PointNeXt应该是256或512）
    if convnext_dim != 768:
        print(f"警告: ConvNeXt 维度是 {convnext_dim}，预期是 768")
        print(f"如果这是错误的，请检查特征文件和数据集的特征匹配逻辑")
    if pointnext_dim not in [256, 512]:
        print(f"警告: PointNeXt 维度是 {pointnext_dim}，预期是 256 或 512")
    
    print(f"最终使用的特征维度: PointNeXt={pointnext_dim}, ConvNeXt={convnext_dim}")
    
    # 创建模型
    model = create_classifier(
        pointnext_dim=pointnext_dim,
        convnext_dim=convnext_dim,
        num_classes=4,  # 4类分类：类别0, 1, 2, 3
        hidden_dim=args.hidden_dim,
        num_fusion_layers=args.num_fusion_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        fusion_mode=args.fusion_mode,
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
    )
    
    # 恢复训练
    start_epoch = 0
    best_val_acc = 0.0
    best_test_acc = 0.0
    use_test_for_best_model = test_loader is not None
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"从epoch {start_epoch}恢复训练")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(start_epoch, args.num_epochs):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        # 测试集评估（如果存在）
        test_metrics = None
        if use_test_for_best_model:
            test_metrics = validate(model, test_loader, criterion, device, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
        if test_metrics is not None:
            writer.add_scalar('Test/Loss', test_metrics['loss'], epoch)
            writer.add_scalar('Test/Accuracy', test_metrics['accuracy'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印统计信息
        print_str = (
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['loss']:.4f}, "
            f"Train Acc={train_metrics['accuracy']:.2f}%, "
            f"Val Loss={val_metrics['loss']:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.2f}%"
        )
        if test_metrics is not None:
            print_str += (
                f", Test Loss={test_metrics['loss']:.4f}, "
                f"Test Acc={test_metrics['accuracy']:.2f}%"
            )
        print(print_str)
        
        # 保存最佳模型（如果测试集存在，基于测试集性能；否则基于验证集性能）
        should_save_best = False
        if use_test_for_best_model and test_metrics is not None:
            # 基于测试集准确率保存最佳模型
            if test_metrics['accuracy'] > best_test_acc:
                best_test_acc = test_metrics['accuracy']
                should_save_best = True
                save_metrics_name = "Test"
                save_metrics = test_metrics
        else:
            # 基于验证集准确率保存最佳模型
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                should_save_best = True
                save_metrics_name = "Val"
                save_metrics = val_metrics
        
        if should_save_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model_config': {
                    'pointnext_dim': pointnext_dim,
                    'convnext_dim': convnext_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_fusion_layers': args.num_fusion_layers,
                    'num_heads': args.num_heads,
                    'dropout': args.dropout,
                    'fusion_mode': args.fusion_mode,
                    'num_classes': 4,
                },
                'args': vars(args),
            }
            if test_metrics is not None:
                checkpoint['test_metrics'] = test_metrics
                if use_test_for_best_model:
                    checkpoint['best_test_acc'] = best_test_acc
            
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"保存最佳模型 (基于{save_metrics_name} Acc: {save_metrics['accuracy']:.2f}%)")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'model_config': {
                    'pointnext_dim': pointnext_dim,
                    'convnext_dim': convnext_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_fusion_layers': args.num_fusion_layers,
                    'num_heads': args.num_heads,
                    'dropout': args.dropout,
                    'fusion_mode': args.fusion_mode,
                    'num_classes': 4,
                },
                'args': vars(args),
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    writer.close()
    
    # 最终评估（使用最佳模型）
    print("\n=== 最终评估（使用最佳模型） ===")
    best_model_path = output_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            saved_epoch = checkpoint.get('epoch', '?')
            print(f"加载最佳模型: {best_model_path} (Epoch {saved_epoch})")
            
            if use_test_for_best_model and 'best_test_acc' in checkpoint:
                print(f"  保存时 - Test Acc: {checkpoint['best_test_acc']:.2f}%")
            else:
                print(f"  保存时 - Val Acc: {checkpoint.get('best_val_acc', 0):.2f}%")
        else:
            model.load_state_dict(checkpoint)
            print(f"加载模型权重: {best_model_path}")
    
    # 在验证集上最终评估
    print("\n验证集最终评估:")
    final_val_metrics = validate(model, val_loader, criterion, device, 0)
    print(
        f"  Accuracy: {final_val_metrics['accuracy']:.2f}%, "
        f"Loss: {final_val_metrics['loss']:.4f}"
    )
    
    # 在测试集上最终评估（如果存在）
    if test_loader is not None:
        print("\n测试集最终评估:")
        final_test_metrics = validate(model, test_loader, criterion, device, 0)
        print(
            f"  Accuracy: {final_test_metrics['accuracy']:.2f}%, "
            f"Loss: {final_test_metrics['loss']:.4f}"
        )
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()

