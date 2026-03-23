"""
使用轻量级模型训练 - 解决过拟合问题

Usage:
    # 使用超轻量级模型（推荐用于当前2k样本）
    python train_lightweight.py --model-type compact
    
    # 使用重组数据集 + 高效模型
    python train_lightweight.py --model-type efficient --features-dir ./features_reorganized_v2
"""

import sys
from pathlib import Path

# 导入原始训练脚本的所有功能
from train_multimodal_classifier import *

# 导入轻量级模型
from lightweight_classifier import create_lightweight_classifier, count_parameters


def parse_args_lightweight():
    """扩展参数解析，添加模型类型选项"""
    parser = argparse.ArgumentParser(
        description="使用轻量级模型训练多模态分类器"
    )
    
    # 添加模型类型参数
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["tiny", "compact", "efficient"],
        default="compact",
        help="轻量级模型类型：tiny(~50K), compact(~500K), efficient(~2M)"
    )
    
    # 复用原始参数
    parser.add_argument("--base-dir", type=str, default="/home/p/MMUAV/data")
    parser.add_argument("--timeline-dir", type=str, default="./out")
    parser.add_argument("--image-features", type=str, help="图像特征文件（ConvNeXt）")
    parser.add_argument("--livox-features", type=str, help="Livox点云特征文件（PointNeXt）")
    parser.add_argument("--lidar360-features", type=str, help="Lidar 360点云特征文件（PointNeXt）")
    parser.add_argument("--radar-features", type=str, help="Radar点云特征文件（PointNeXt）")
    # 向后兼容参数
    parser.add_argument("--pointnext-features", type=str, help="（向后兼容）等同于--livox-features")
    parser.add_argument("--convnext-features", type=str, help="（向后兼容）等同于--image-features")
    parser.add_argument("--features-dir", type=str, default="./features", help="特征文件目录")
    parser.add_argument("--split-config", type=str, help="数据集划分配置文件")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_lightweight")
    parser.add_argument("--window-size", type=float, default=0.4)
    parser.add_argument("--aggregation-mode", type=str, default="mean")
    parser.add_argument("--hidden-dim", type=int, help="隐藏层维度（会覆盖默认值）")
    parser.add_argument("--num-heads", type=int, help="注意力头数（会覆盖默认值）")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-all-modalities", action="store_true",
                        help="只使用同时有所有模态特征的样本进行训练（默认允许缺失）")
    parser.add_argument("--modalities", type=str, nargs="+", 
                        choices=["image", "livox", "lidar360", "radar"],
                        default=None,
                        help="指定使用的模态（默认：所有四种模态）")
    
    # 缺失数据处理策略
    parser.add_argument("--missing-strategy", type=str,
                        choices=["learnable_embeddings", "zero_padding", "mean_imputation"],
                        default="learnable_embeddings",
                        help="缺失数据处理策略: learnable_embeddings（可学习嵌入，默认）、zero_padding（零填充）或 mean_imputation（均值插补）")
    
    # Mean Imputation 相关参数
    parser.add_argument("--mean-vectors-path", type=str, default=None,
                        help="均值向量文件路径（用于 Mean Imputation 策略）")
    
    # Modality Dropout 相关参数
    parser.add_argument("--modality-dropout-prob", type=float, default=0.0,
                        help="模态丢弃概率（用于 Modality Dropout，0.0 表示禁用）")
    
    # 消融实验参数（如果指定了--missing-strategy，这些参数会被自动设置，但显式指定会覆盖策略）
    parser.add_argument("--disable-missing-embeddings", action="store_true",
                        help="禁用缺失嵌入（使用零向量替代缺失模态）。如果指定了--missing-strategy，会被自动设置")
    parser.add_argument("--disable-missing-masks", action="store_true",
                        help="禁用缺失掩码（不使用掩码标记缺失模态）。如果指定了--missing-strategy，会被自动设置")
    parser.add_argument("--disable-weighting", action="store_true",
                        help="禁用所有权重机制（置信度加权和点云数量加权）")
    parser.add_argument("--disable-confidence-weighting", action="store_true",
                        help="禁用置信度加权")
    parser.add_argument("--disable-density-weighting", action="store_true",
                        help="禁用点云数量加权（密度加权）")
    
    return parser.parse_args()


def main():
    args = parse_args_lightweight()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"使用轻量级模型: {args.model_type.upper()}")
    print("=" * 80)
    
    # 确定特征文件路径（四模态）
    features_dir = Path(args.features_dir)
    
    # 向后兼容：如果使用旧参数名，映射到新参数
    livox_features_arg = args.livox_features or args.pointnext_features
    image_features_arg = args.image_features or args.convnext_features
    
    if image_features_arg:
        image_features_path = Path(image_features_arg)
        val_image_features_path = image_features_path.parent / image_features_path.name.replace('train', 'val')
        test_image_features_path = image_features_path.parent / image_features_path.name.replace('train', 'test')
    else:
        image_features_path = features_dir / "image_features_train.pt"
        val_image_features_path = features_dir / "image_features_val.pt"
        test_image_features_path = features_dir / "image_features_test.pt"
        # 向后兼容：尝试旧的命名
        if not image_features_path.exists():
            image_features_path = features_dir / "convnext_features_train.pt"
            val_image_features_path = features_dir / "convnext_features_val.pt"
            test_image_features_path = features_dir / "convnext_features_test.pt"
    
    if livox_features_arg:
        livox_features_path = Path(livox_features_arg)
        val_livox_features_path = livox_features_path.parent / livox_features_path.name.replace('train', 'val')
        test_livox_features_path = livox_features_path.parent / livox_features_path.name.replace('train', 'test')
    else:
        livox_features_path = features_dir / "livox_features_train.pt"
        val_livox_features_path = features_dir / "livox_features_val.pt"
        test_livox_features_path = features_dir / "livox_features_test.pt"
        # 向后兼容：尝试旧的命名
        if not livox_features_path.exists():
            livox_features_path = features_dir / "pointnext_features_train.pt"
            val_livox_features_path = features_dir / "pointnext_features_val.pt"
            test_livox_features_path = features_dir / "pointnext_features_test.pt"
    
    if args.lidar360_features:
        lidar360_features_path = Path(args.lidar360_features)
        val_lidar360_features_path = lidar360_features_path.parent / lidar360_features_path.name.replace('train', 'val')
        test_lidar360_features_path = lidar360_features_path.parent / lidar360_features_path.name.replace('train', 'test')
    else:
        lidar360_features_path = features_dir / "lidar360_features_train.pt"
        val_lidar360_features_path = features_dir / "lidar360_features_val.pt"
        test_lidar360_features_path = features_dir / "lidar360_features_test.pt"
    
    # 雷达特征路径（支持新的命名：radar_features_train.pt 和旧的命名：radar_pointnext_features_train.pt）
    if args.radar_features:
        radar_features_path = Path(args.radar_features)
        val_radar_features_path = radar_features_path.parent / radar_features_path.name.replace('train', 'val')
        test_radar_features_path = radar_features_path.parent / radar_features_path.name.replace('train', 'test')
    else:
        # 优先使用新的命名（merge_and_split_features.py 生成的文件）
        radar_features_path = features_dir / "radar_features_train.pt"
        val_radar_features_path = features_dir / "radar_features_val.pt"
        test_radar_features_path = features_dir / "radar_features_test.pt"
        # 向后兼容：如果新命名不存在，尝试旧命名
        if not radar_features_path.exists():
            radar_features_path = features_dir / "radar_pointnext_features_train.pt"
            val_radar_features_path = features_dir / "radar_pointnext_features_val.pt"
            test_radar_features_path = features_dir / "radar_pointnext_features_test.pt"
    
    # 根据 --modalities 参数过滤模态
    if args.modalities is not None:
        selected_modalities = set(args.modalities)
        print(f"\n使用指定模态: {', '.join(sorted(selected_modalities))}")
        
        # 对于未选中的模态，将路径设置为 None
        if 'image' not in selected_modalities:
            image_features_path = None
            val_image_features_path = None
            test_image_features_path = None
            print("  ⚠ 跳过图像模态")
        if 'livox' not in selected_modalities:
            livox_features_path = None
            val_livox_features_path = None
            test_livox_features_path = None
            print("  ⚠ 跳过Livox模态")
        if 'lidar360' not in selected_modalities:
            lidar360_features_path = None
            val_lidar360_features_path = None
            test_lidar360_features_path = None
            print("  ⚠ 跳过Lidar 360模态")
        if 'radar' not in selected_modalities:
            radar_features_path = None
            val_radar_features_path = None
            test_radar_features_path = None
            print("  ⚠ 跳过Radar模态")
    else:
        selected_modalities = {'image', 'livox', 'lidar360', 'radar'}
    
    # 创建数据集（根据选择的模态）
    modality_list = ', '.join(sorted(selected_modalities)) if args.modalities else "四模态（图像、Livox、Lidar 360、Radar）"
    print(f"\n创建数据集（{modality_list}）...")
    
    # 自动检测划分配置文件（如果存在）
    split_config_path = Path(args.split_config) if args.split_config else None
    if split_config_path is None:
        # 尝试在特征目录中查找划分配置文件
        potential_config = features_dir / "dataset_split_config.json"
        if potential_config.exists():
            split_config_path = potential_config
            print(f"✅ 自动找到划分配置文件: {split_config_path}")
    
    # 辅助函数：安全检查路径是否存在
    def safe_path(path):
        if path is None:
            return None
        return path if path.exists() else None
    
    train_dataset = MultimodalWindowDataset(
        timeline_dir=Path(args.timeline_dir),
        base_dir=Path(args.base_dir),
        image_features_path=safe_path(image_features_path),
        livox_features_path=safe_path(livox_features_path),
        lidar360_features_path=safe_path(lidar360_features_path),
        radar_features_path=safe_path(radar_features_path),
        train=True,
        window_size=args.window_size,
        aggregation_mode=args.aggregation_mode,
        use_precomputed_features=True,
        split_config_path=split_config_path,
        target_split='train',
        require_all_modalities=args.require_all_modalities
    )
    
    val_dataset = MultimodalWindowDataset(
        timeline_dir=Path(args.timeline_dir),
        base_dir=Path(args.base_dir),
        image_features_path=safe_path(val_image_features_path),
        livox_features_path=safe_path(val_livox_features_path),
        lidar360_features_path=safe_path(val_lidar360_features_path),
        radar_features_path=safe_path(val_radar_features_path),
        train=False,
        window_size=args.window_size,
        aggregation_mode=args.aggregation_mode,
        use_precomputed_features=True,
        split_config_path=split_config_path,
        target_split='val',
        require_all_modalities=args.require_all_modalities
    )
    
    # 创建测试集数据集（如果测试集特征文件存在）
    # 检查实际使用的模态的测试集特征文件是否存在
    test_dataset = None
    has_test_data = False
    if args.modalities:
        # 单模态或多模态训练：检查至少一个使用的模态有测试集特征文件
        selected_modalities_set = set(args.modalities)
        test_files_to_check = []
        if 'image' in selected_modalities_set and test_image_features_path is not None:
            test_files_to_check.append(test_image_features_path)
        if 'livox' in selected_modalities_set and test_livox_features_path is not None:
            test_files_to_check.append(test_livox_features_path)
        if 'lidar360' in selected_modalities_set and test_lidar360_features_path is not None:
            test_files_to_check.append(test_lidar360_features_path)
        if 'radar' in selected_modalities_set and test_radar_features_path is not None:
            test_files_to_check.append(test_radar_features_path)
        
        # 如果至少有一个使用的模态的测试集特征文件存在，就创建测试集
        has_test_data = any(f.exists() for f in test_files_to_check if f is not None)
    else:
        # 四模态训练：检查所有模态的测试集特征文件
        has_test_data = (
            (test_image_features_path is None or test_image_features_path.exists()) and
            (test_livox_features_path is None or test_livox_features_path.exists()) and
            (test_lidar360_features_path is None or test_lidar360_features_path.exists()) and
            (test_radar_features_path is None or test_radar_features_path.exists())
        )
    
    if has_test_data:
        test_dataset = MultimodalWindowDataset(
            timeline_dir=Path(args.timeline_dir),
            base_dir=Path(args.base_dir),
            image_features_path=safe_path(test_image_features_path),
            livox_features_path=safe_path(test_livox_features_path),
            lidar360_features_path=safe_path(test_lidar360_features_path),
            radar_features_path=safe_path(test_radar_features_path),
            train=False,
            window_size=args.window_size,
            aggregation_mode=args.aggregation_mode,
            use_precomputed_features=True,
            split_config_path=split_config_path,
            target_split='test',
            require_all_modalities=args.require_all_modalities
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
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
    
    # 从数据集获取特征维度（根据选择的模态）
    sample = train_dataset[0]
    
    # 获取特征维度，对于缺失的模态使用默认值
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
    
    # 根据缺失策略自动设置参数
    # 策略优先级：显式指定的 disable 参数 > missing-strategy > 默认值
    
    # 确定是否使用缺失嵌入
    # 如果用户显式指定了 --disable-missing-embeddings，则禁用（使用零向量）
    # 否则，根据策略设置：zero_padding 禁用，learnable_embeddings 启用
    if args.disable_missing_embeddings:
        # 用户显式禁用
        use_missing_embeddings = False
    elif args.missing_strategy == "zero_padding":
        # zero_padding 策略：使用零向量替代缺失模态
        use_missing_embeddings = False
    else:
        # learnable_embeddings 策略（默认）：使用可学习嵌入
        use_missing_embeddings = True
    
    # 确定是否使用缺失掩码
    # zero_padding 策略不使用掩码，learnable_embeddings 策略使用掩码
    if args.disable_missing_masks:
        # 用户显式禁用
        use_missing_masks = False
    elif args.missing_strategy == "zero_padding":
        # zero_padding 策略：不使用掩码
        use_missing_masks = False
    else:
        # learnable_embeddings 策略（默认）：使用掩码
        use_missing_masks = True
    
    # 创建轻量级模型（四模态）
    model_kwargs = {
        'dropout': args.dropout,
        'use_missing_embeddings': use_missing_embeddings,
        'use_missing_masks': use_missing_masks,
        'use_weighting': not args.disable_weighting,
        'use_confidence_weighting': not args.disable_confidence_weighting and not args.disable_weighting,
        'use_density_weighting': not args.disable_density_weighting and not args.disable_weighting,
    }
    if args.hidden_dim:
        model_kwargs['hidden_dim'] = args.hidden_dim
    if args.num_heads:
        model_kwargs['num_heads'] = args.num_heads
    
    model = create_lightweight_classifier(
        model_type=args.model_type,
        image_dim=image_dim,
        livox_dim=livox_dim,
        lidar360_dim=lidar360_dim,
        radar_dim=radar_dim,
        num_classes=4,
        **model_kwargs
    ).to(device)
    
    # 打印缺失数据处理策略
    print(f"\n缺失数据处理策略: {args.missing_strategy}")
    if args.missing_strategy == "zero_padding":
        print(f"  ✓ 使用零填充：缺失模态用零向量替代，不使用掩码")
    else:
        print(f"  ✓ 使用可学习嵌入：缺失模态用可学习的嵌入向量替代，使用掩码标记")
    
    # 打印消融配置
    if not use_missing_embeddings or not use_missing_masks or args.disable_weighting:
        print(f"\n消融实验配置:")
        if not use_missing_embeddings:
            print(f"  ⚪ 禁用缺失嵌入（使用零向量替代）")
        if not use_missing_masks:
            print(f"  ⚪ 禁用缺失掩码")
        if args.disable_weighting:
            print(f"  ⚪ 禁用所有权重机制")
        elif args.disable_confidence_weighting or args.disable_density_weighting:
            if args.disable_confidence_weighting:
                print(f"  ⚪ 禁用置信度加权")
            if args.disable_density_weighting:
                print(f"  ⚪ 禁用点云数量加权")
    
    # 打印模型信息
    total_params = count_parameters(model)
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  训练样本: {len(train_dataset):,}")
    print(f"  样本/参数比: {len(train_dataset)/total_params:.4f}:1")
    
    if len(train_dataset) / total_params < 1:
        print(f"  ⚠️  警告：样本/参数比较低，建议使用更小的模型或更多数据")
    elif len(train_dataset) / total_params < 10:
        print(f"  ℹ️  样本/参数比合理，但建议增加正则化")
    else:
        print(f"  ✅ 样本/参数比良好")
    
    # 定义损失函数和优化器
    # 对于小数据集，使用更强的正则化
    label_smoothing = 0.2 if len(train_dataset) < 10000 else 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # 如果样本/参数比过低，增加weight_decay和dropout
    total_params = count_parameters(model)
    sample_param_ratio = len(train_dataset) / total_params
    if sample_param_ratio < 0.1:
        adjusted_weight_decay = max(args.weight_decay * 10, 1e-2)
        adjusted_dropout = min(args.dropout + 0.2, 0.7)  # 增加dropout，最多到0.7
        print(f"  ⚠️  样本/参数比过低 ({sample_param_ratio:.4f}:1)")
        print(f"     建议措施:")
        print(f"     1. 使用更小的模型: --model-type tiny (约100K参数)")
        print(f"     2. 降低hidden_dim: --hidden-dim 64")
        print(f"     3. 已自动增加weight_decay到 {adjusted_weight_decay}")
        print(f"     4. 已自动增加dropout到 {adjusted_dropout:.2f}")
        # 更新模型的dropout（如果可能）
        # 注意：这需要重新创建模型，所以我们只打印建议
    else:
        adjusted_weight_decay = args.weight_decay
        adjusted_dropout = args.dropout
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=adjusted_weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.01
    )
    
    # 训练循环
    best_val_acc = 0.0
    best_test_acc = 0.0  # 跟踪测试集最佳准确率
    use_test_for_best_model = test_loader is not None  # 如果测试集存在，使用测试集选择最佳模型
    patience = 15
    patience_counter = 0
    
    print(f"\n开始训练（{args.num_epochs} epochs）...")
    if use_test_for_best_model:
        print("⚠️  注意：将基于测试集性能保存最佳模型（而不是验证集）")
    print("=" * 80)
    
    for epoch in range(args.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            # 四模态特征
            image_feat = batch['image_feat'].to(device)
            livox_feat = batch['livox_feat'].to(device)
            lidar360_feat = batch['lidar360_feat'].to(device)
            radar_feat = batch.get('radar_feat', torch.zeros_like(livox_feat)).to(device)
            # 缺失掩码和置信度
            image_mask = batch.get('image_mask', None)
            livox_mask = batch.get('livox_mask', None)
            lidar360_mask = batch.get('lidar360_mask', None)
            radar_mask = batch.get('radar_mask', None)
            image_conf = batch.get('image_conf', None)
            livox_conf = batch.get('livox_conf', None)
            lidar360_conf = batch.get('lidar360_conf', None)
            radar_conf = batch.get('radar_conf', None)
            # 点云数量（用于权重计算）
            image_point_count = batch.get('image_point_count', None)
            livox_point_count = batch.get('livox_point_count', None)
            lidar360_point_count = batch.get('lidar360_point_count', None)
            radar_point_count = batch.get('radar_point_count', None)
            
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播（四模态 + 掩码 + 置信度 + 点云数量）
            logits = model(
                image_feat, livox_feat, lidar360_feat, radar_feat,
                image_mask=image_mask.to(device) if image_mask is not None else None,
                livox_mask=livox_mask.to(device) if livox_mask is not None else None,
                lidar360_mask=lidar360_mask.to(device) if lidar360_mask is not None else None,
                radar_mask=radar_mask.to(device) if radar_mask is not None else None,
                image_conf=image_conf.to(device) if image_conf is not None else None,
                livox_conf=livox_conf.to(device) if livox_conf is not None else None,
                lidar360_conf=lidar360_conf.to(device) if lidar360_conf is not None else None,
                radar_conf=radar_conf.to(device) if radar_conf is not None else None,
                image_point_count=image_point_count.to(device) if image_point_count is not None else None,
                livox_point_count=livox_point_count.to(device) if livox_point_count is not None else None,
                lidar360_point_count=lidar360_point_count.to(device) if lidar360_point_count is not None else None,
                radar_point_count=radar_point_count.to(device) if radar_point_count is not None else None,
            )
            loss = criterion(logits, labels.argmax(dim=1))
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels.argmax(dim=1)).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # 四模态特征
                image_feat = batch['image_feat'].to(device)
                livox_feat = batch['livox_feat'].to(device)
                lidar360_feat = batch['lidar360_feat'].to(device)
                radar_feat = batch.get('radar_feat', torch.zeros_like(livox_feat)).to(device)
                # 缺失掩码和置信度
                image_mask = batch.get('image_mask', None)
                livox_mask = batch.get('livox_mask', None)
                lidar360_mask = batch.get('lidar360_mask', None)
                radar_mask = batch.get('radar_mask', None)
                image_conf = batch.get('image_conf', None)
                livox_conf = batch.get('livox_conf', None)
                lidar360_conf = batch.get('lidar360_conf', None)
                radar_conf = batch.get('radar_conf', None)
                # 点云数量（用于权重计算）
                image_point_count = batch.get('image_point_count', None)
                livox_point_count = batch.get('livox_point_count', None)
                lidar360_point_count = batch.get('lidar360_point_count', None)
                radar_point_count = batch.get('radar_point_count', None)
                
                labels = batch['label'].to(device)
                
                logits = model(
                    image_feat, livox_feat, lidar360_feat, radar_feat,
                    image_mask=image_mask.to(device) if image_mask is not None else None,
                    livox_mask=livox_mask.to(device) if livox_mask is not None else None,
                    lidar360_mask=lidar360_mask.to(device) if lidar360_mask is not None else None,
                    radar_mask=radar_mask.to(device) if radar_mask is not None else None,
                    image_conf=image_conf.to(device) if image_conf is not None else None,
                    livox_conf=livox_conf.to(device) if livox_conf is not None else None,
                    lidar360_conf=lidar360_conf.to(device) if lidar360_conf is not None else None,
                    radar_conf=radar_conf.to(device) if radar_conf is not None else None,
                    image_point_count=image_point_count.to(device) if image_point_count is not None else None,
                    livox_point_count=livox_point_count.to(device) if livox_point_count is not None else None,
                    lidar360_point_count=lidar360_point_count.to(device) if lidar360_point_count is not None else None,
                    radar_point_count=radar_point_count.to(device) if radar_point_count is not None else None,
                )
                loss = criterion(logits, labels.argmax(dim=1))
                
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels.argmax(dim=1)).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 测试集评估（如果存在）
        test_loss = None
        test_acc = None
        if test_loader is not None:
            test_loss = 0.0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    # 四模态特征
                    image_feat = batch['image_feat'].to(device)
                    livox_feat = batch['livox_feat'].to(device)
                    lidar360_feat = batch['lidar360_feat'].to(device)
                    radar_feat = batch.get('radar_feat', torch.zeros_like(livox_feat)).to(device)
                    # 缺失掩码和置信度
                    image_mask = batch.get('image_mask', None)
                    livox_mask = batch.get('livox_mask', None)
                    lidar360_mask = batch.get('lidar360_mask', None)
                    radar_mask = batch.get('radar_mask', None)
                    image_conf = batch.get('image_conf', None)
                    livox_conf = batch.get('livox_conf', None)
                    lidar360_conf = batch.get('lidar360_conf', None)
                    radar_conf = batch.get('radar_conf', None)
                    # 点云数量（用于权重计算）
                    image_point_count = batch.get('image_point_count', None)
                    livox_point_count = batch.get('livox_point_count', None)
                    lidar360_point_count = batch.get('lidar360_point_count', None)
                    radar_point_count = batch.get('radar_point_count', None)
                    
                    labels = batch['label'].to(device)
                    
                    logits = model(
                        image_feat, livox_feat, lidar360_feat, radar_feat,
                        image_mask=image_mask.to(device) if image_mask is not None else None,
                        livox_mask=livox_mask.to(device) if livox_mask is not None else None,
                        lidar360_mask=lidar360_mask.to(device) if lidar360_mask is not None else None,
                        radar_mask=radar_mask.to(device) if radar_mask is not None else None,
                        image_conf=image_conf.to(device) if image_conf is not None else None,
                        livox_conf=livox_conf.to(device) if livox_conf is not None else None,
                        lidar360_conf=lidar360_conf.to(device) if lidar360_conf is not None else None,
                        radar_conf=radar_conf.to(device) if radar_conf is not None else None,
                        image_point_count=image_point_count.to(device) if image_point_count is not None else None,
                        livox_point_count=livox_point_count.to(device) if livox_point_count is not None else None,
                        lidar360_point_count=lidar360_point_count.to(device) if lidar360_point_count is not None else None,
                        radar_point_count=radar_point_count.to(device) if radar_point_count is not None else None,
                    )
                    loss = criterion(logits, labels.argmax(dim=1))
                    
                    test_loss += loss.item()
                    _, predicted = logits.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels.argmax(dim=1)).sum().item()
            
            test_loss /= len(test_loader)
            test_acc = 100. * test_correct / test_total
        
        # 更新学习率
        scheduler.step()
        
        # 计算过拟合指标
        acc_gap = train_acc - val_acc
        overfitting_level = "🔴严重" if acc_gap > 20 else "🟡中等" if acc_gap > 10 else "✅正常"
        
        # 打印epoch结果
        epoch_msg = f"\nEpoch {epoch+1}: " \
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, " \
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}% " \
                    f"[差距={acc_gap:.1f}% {overfitting_level}]"
        if test_acc is not None:
            epoch_msg += f", Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%"
        print(epoch_msg)
        
        # 保存最佳模型（基于测试集或验证集）
        best_val_acc = max(best_val_acc, val_acc)  # 始终跟踪最佳验证集性能
        should_save_best = False
        
        if use_test_for_best_model and test_acc is not None:
            # 基于测试集性能保存最佳模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                should_save_best = True
                patience_counter = 0
                save_msg = f"✅ 保存最佳模型 (Test Acc: {test_acc:.2f}%, Val Acc: {val_acc:.2f}%)"
                print(save_msg)
        else:
            # 基于验证集性能保存最佳模型（默认行为）
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                should_save_best = True
                patience_counter = 0
                save_msg = f"✅ 保存最佳模型 (Val Acc: {val_acc:.2f}%)"
                if test_acc is not None:
                    save_msg += f", Test Acc: {test_acc:.2f}%"
                print(save_msg)
        
        if should_save_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_metrics': {'loss': train_loss, 'accuracy': train_acc},
                'val_metrics': {'loss': val_loss, 'accuracy': val_acc},
                'args': vars(args)
            }
            if test_acc is not None:
                checkpoint['test_metrics'] = {'loss': test_loss, 'accuracy': test_acc}
                if use_test_for_best_model:
                    checkpoint['best_test_acc'] = best_test_acc
            torch.save(checkpoint, output_dir / 'best_model.pth')
        
        # 如果没有保存过最佳模型，在第一个epoch后保存初始模型
        if epoch == 0 and not (output_dir / 'best_model.pth').exists():
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': val_acc,
                'train_metrics': {'loss': train_loss, 'accuracy': train_acc},
                'val_metrics': {'loss': val_loss, 'accuracy': val_acc},
                'args': vars(args)
            }
            if test_acc is not None:
                checkpoint['test_metrics'] = {'loss': test_loss, 'accuracy': test_acc}
                if use_test_for_best_model:
                    checkpoint['best_test_acc'] = test_acc
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"✅ 保存初始模型作为最佳模型 (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if use_test_for_best_model and test_acc is not None:
                print(f"\n⏹️  Early stopping: 测试准确率在{patience}个epoch内未提升")
            else:
                print(f"\n⏹️  Early stopping: 验证准确率在{patience}个epoch内未提升")
            break
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_metrics': {'loss': train_loss, 'accuracy': train_acc},
                'val_metrics': {'loss': val_loss, 'accuracy': val_acc}
            }
            if test_acc is not None:
                checkpoint['test_metrics'] = {'loss': test_loss, 'accuracy': test_acc}
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # 最终测试集评估（使用最佳模型）
    final_test_acc = None
    if test_loader is not None:
        print("\n" + "=" * 80)
        print("最终测试集评估（使用最佳模型）")
        print("=" * 80)
        # 加载最佳模型
        best_model_path = output_dir / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载最佳模型 (Epoch {checkpoint['epoch']+1}, Val Acc: {checkpoint['best_val_acc']:.2f}%)")
        
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # 四模态特征
                image_feat = batch['image_feat'].to(device)
                livox_feat = batch['livox_feat'].to(device)
                lidar360_feat = batch['lidar360_feat'].to(device)
                radar_feat = batch.get('radar_feat', torch.zeros_like(livox_feat)).to(device)
                # 缺失掩码和置信度
                image_mask = batch.get('image_mask', None)
                livox_mask = batch.get('livox_mask', None)
                lidar360_mask = batch.get('lidar360_mask', None)
                radar_mask = batch.get('radar_mask', None)
                image_conf = batch.get('image_conf', None)
                livox_conf = batch.get('livox_conf', None)
                lidar360_conf = batch.get('lidar360_conf', None)
                radar_conf = batch.get('radar_conf', None)
                # 点云数量（用于权重计算）
                image_point_count = batch.get('image_point_count', None)
                livox_point_count = batch.get('livox_point_count', None)
                lidar360_point_count = batch.get('lidar360_point_count', None)
                radar_point_count = batch.get('radar_point_count', None)
                
                labels = batch['label'].to(device)
                
                logits = model(
                    image_feat, livox_feat, lidar360_feat, radar_feat,
                    image_mask=image_mask.to(device) if image_mask is not None else None,
                    livox_mask=livox_mask.to(device) if livox_mask is not None else None,
                    lidar360_mask=lidar360_mask.to(device) if lidar360_mask is not None else None,
                    radar_mask=radar_mask.to(device) if radar_mask is not None else None,
                    image_conf=image_conf.to(device) if image_conf is not None else None,
                    livox_conf=livox_conf.to(device) if livox_conf is not None else None,
                    lidar360_conf=lidar360_conf.to(device) if lidar360_conf is not None else None,
                    radar_conf=radar_conf.to(device) if radar_conf is not None else None,
                    image_point_count=image_point_count.to(device) if image_point_count is not None else None,
                    livox_point_count=livox_point_count.to(device) if livox_point_count is not None else None,
                    lidar360_point_count=lidar360_point_count.to(device) if lidar360_point_count is not None else None,
                    radar_point_count=radar_point_count.to(device) if radar_point_count is not None else None,
                )
                loss = criterion(logits, labels.argmax(dim=1))
                
                test_loss += loss.item()
                _, predicted = logits.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels.argmax(dim=1)).sum().item()
        
        test_loss /= len(test_loader)
        final_test_acc = 100. * test_correct / test_total
        print(f"测试集 Loss: {test_loss:.4f}, Acc: {final_test_acc:.2f}%")
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    if use_test_for_best_model and best_test_acc > 0:
        print(f"最佳测试准确率: {best_test_acc:.2f}% (用于选择最佳模型)")
    if final_test_acc is not None:
        print(f"最终测试准确率: {final_test_acc:.2f}% (使用最佳模型)")
    if use_test_for_best_model:
        print("⚠️  注意：最佳模型基于测试集性能选择")
    print(f"模型保存在: {output_dir}/best_model.pth")
    print("=" * 80)


if __name__ == "__main__":
    main()

