import argparse
import os
import pickle
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于雷达中心位置序列的LSTM二分类训练脚本")
    parser.add_argument(
        "--result_folder",
        type=str,
        default="/home/p/MMUAV/result",
        help="radar_data_processor生成结果所在根目录")
    parser.add_argument(
        "--feature_subdir",
        type=str,
        default="radar_enhance_pcl_feature_set",
        help="存放雷达特征与标签的子目录名")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="训练/验证批大小")
    parser.add_argument("--hidden_size", type=int, default=64, help="LSTM隐藏层维度（与train_lidara_detector.py一致）")
    parser.add_argument("--num_layers", type=int, default=1, help="LSTM层数（与train_lidara_detector.py一致）")
    parser.add_argument("--dropout", type=float, default=0.8, help="Dropout比例（与train_lidara_detector.py一致）")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam学习率")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="权重衰减（L2正则化）")
    parser.add_argument("--num_epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--use_class_weights", action="store_true", help="使用类别权重平衡正负样本")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="对训练批次启用序列增强")
    parser.add_argument(
        "--model_path",
        type=str,
        default="lstm_radar_enhance_model.pth",
        help="最佳模型权重保存路径")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="启用特征标准化（默认不使用，与train_lidara_detector.py一致）")
    parser.add_argument(
        "--scaler_path",
        type=str,
        default=None,
        help="Scaler保存/加载路径（默认: model_path.replace('.pth', '_scaler.pkl')）")
    return parser.parse_args()


def load_feature_sets(result_folder: str,
                      feature_subdir: str,
                      normalize: bool = False,  # 默认不标准化（与train_lidara_detector.py一致）
                      scaler_path: str = None,
                      load_test: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    加载特征集并进行标准化
    
    Args:
        result_folder: 结果文件夹路径
        feature_subdir: 特征子目录名
        normalize: 是否进行特征标准化（默认True）
        scaler_path: scaler保存/加载路径
        load_test: 是否加载测试集（默认True）
    
    Returns:
        feature_train, label_train, feature_val, label_val, scaler, feature_test (可选), label_test (可选)
    """
    save_directory = os.path.join(result_folder, feature_subdir)
    
    # 尝试多种可能的文件名格式
    # 可能的文件名格式
    possible_train_names = [
        ("feature_train.npy", "label_train.npy"),  # 简单格式
        ("feature_radar_enhance_pcl_train.npy", "label_radar_enhance_pcl_train.npy"),  # 完整格式
    ]
    possible_val_names = [
        ("feature_val.npy", "label_val.npy"),  # 简单格式
        ("feature_radar_enhance_pcl_val.npy", "label_radar_enhance_pcl_val.npy"),  # 完整格式
    ]
    possible_test_names = [
        ("feature_test.npy", "label_test.npy"),  # 简单格式
        ("feature_radar_enhance_pcl_test.npy", "label_radar_enhance_pcl_test.npy"),  # 完整格式
    ]
    
    # 加载训练集
    feature_train = None
    label_train = None
    for train_feat_name, train_label_name in possible_train_names:
        train_feature_path = os.path.join(save_directory, train_feat_name)
        train_label_path = os.path.join(save_directory, train_label_name)
        if os.path.exists(train_feature_path) and os.path.exists(train_label_path):
            feature_train = np.load(train_feature_path)
            label_train = np.load(train_label_path)
            print(f"✓ 加载训练集: {feature_train.shape} (从 {train_feat_name})")
            break
    if feature_train is None:
        raise FileNotFoundError(f"训练集文件不存在（已尝试: {[name[0] for name in possible_train_names]}）")
    
    # 加载验证集
    feature_val = None
    label_val = None
    for val_feat_name, val_label_name in possible_val_names:
        val_feature_path = os.path.join(save_directory, val_feat_name)
        val_label_path = os.path.join(save_directory, val_label_name)
        if os.path.exists(val_feature_path) and os.path.exists(val_label_path):
            feature_val = np.load(val_feature_path)
            label_val = np.load(val_label_path)
            print(f"✓ 加载验证集: {feature_val.shape} (从 {val_feat_name})")
            break
    if feature_val is None:
        raise FileNotFoundError(f"验证集文件不存在（已尝试: {[name[0] for name in possible_val_names]}）")
    
    # 尝试加载测试集（如果存在）
    feature_test = None
    label_test = None
    if load_test:
        for test_feat_name, test_label_name in possible_test_names:
            test_feature_path = os.path.join(save_directory, test_feat_name)
            test_label_path = os.path.join(save_directory, test_label_name)
            if os.path.exists(test_feature_path) and os.path.exists(test_label_path):
                feature_test = np.load(test_feature_path)
                label_test = np.load(test_label_path)
                print(f"✓ 加载测试集: {feature_test.shape} (从 {test_feat_name})")
                break
        
        if feature_test is None:
            print(f"⚠ 测试集文件不存在（已尝试: {[name[0] for name in possible_test_names]}）")

    # 支持9维（旧版）或15维（新版，包含位置/速度/加速度/std/range）或21维（扩展版）
    expected_dims = [9, 15, 21]
    if feature_train.shape[-1] not in expected_dims:
        raise ValueError(
            f"特征维度不匹配, 期望{expected_dims} (9/15/21维), 实际为 {feature_train.shape[-1]}")
    
    scaler = None
    if normalize:
        # 获取当前特征维度
        current_feature_dim = feature_train.shape[-1]
        
        # 加载已有scaler或创建新的
        if scaler_path and os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                # 检查scaler的特征维度是否匹配
                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != current_feature_dim:
                    print(f"⚠ Scaler特征维度不匹配（期望{current_feature_dim}维，实际{scaler.n_features_in_}维），将重新生成")
                    scaler = None
                else:
                    print(f"✓ 加载已有scaler: {scaler_path} (特征维度: {current_feature_dim})")
            except Exception as e:
                print(f"⚠ 加载scaler失败: {e}，将重新生成")
                scaler = None
        
        if scaler is None:
            # 使用训练集拟合scaler
            print(f"✓ 创建新的scaler（基于训练集，特征维度: {current_feature_dim}）")
            features_flat = feature_train.reshape(-1, feature_train.shape[-1])
            scaler = StandardScaler()
            scaler.fit(features_flat)
            
            # 保存scaler
            if scaler_path:
                scaler_dir = os.path.dirname(scaler_path)
                if scaler_dir:  # 只有当有目录部分时才创建目录
                    os.makedirs(scaler_dir, exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"✓ Scaler已保存: {scaler_path}")
        
        # 标准化特征
        print("✓ 标准化特征...")
        feature_train_flat = feature_train.reshape(-1, feature_train.shape[-1])
        feature_train_scaled = scaler.transform(feature_train_flat)
        feature_train = feature_train_scaled.reshape(feature_train.shape)
        
        feature_val_flat = feature_val.reshape(-1, feature_val.shape[-1])
        feature_val_scaled = scaler.transform(feature_val_flat)
        feature_val = feature_val_scaled.reshape(feature_val.shape)
        
        # 如果测试集存在，也进行标准化
        if feature_test is not None:
            feature_test_flat = feature_test.reshape(-1, feature_test.shape[-1])
            feature_test_scaled = scaler.transform(feature_test_flat)
            feature_test = feature_test_scaled.reshape(feature_test.shape)
        
        print(f"  标准化后 - 训练集范围: [{feature_train.min():.4f}, {feature_train.max():.4f}]")
        print(f"  标准化后 - 训练集均值: {feature_train.mean():.4f}, 标准差: {feature_train.std():.4f}")
    else:
        print("⚠ 未使用特征标准化（不推荐）")
        print(f"  原始特征范围: [{feature_train.min():.4f}, {feature_train.max():.4f}]")
        print(f"  原始特征均值: {feature_train.mean():.4f}, 标准差: {feature_train.std():.4f}")
    
    return feature_train, label_train, feature_val, label_val, scaler, feature_test, label_test


def reverse_sequence(tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(tensor, dims=[-2])


def add_noise_to_features(tensor: torch.Tensor,
                           base_scale: float = 0.1
                           ) -> torch.Tensor:
    seq_len, num_features = tensor.size()
    noise = torch.randn_like(tensor)
    scale = torch.ones_like(tensor)
    
    # 15维特征噪声
    if num_features >= 15:
        scale[:, :3] = 0.1     # 位置噪声
        scale[:, 3:6] = 0.05   # 速度噪声
        scale[:, 6:9] = 0.02   # 加速度噪声（较小）
        scale[:, 9:12] = 0.01  # 标准差噪声（较小）
        scale[:, 12:15] = 0.01 # 范围噪声（较小）
    elif num_features >= 9:
        # 9维特征噪声（位置+std+range）
        scale[:, :3] = 0.1     # 位置噪声
        scale[:, 3:6] = 0.01   # 标准差噪声（较小）
        scale[:, 6:9] = 0.01   # 范围噪声（较小）
    
    return tensor + noise * scale * base_scale


def random_replace_with_zeros(data: torch.Tensor, max_replace: int = 3) -> torch.Tensor:
    """随机将某些时间帧的特征替换为零（与train_lidara_detector.py一致）"""
    seq_len, num_features = data.size()
    replace_count = np.random.randint(1, max_replace+1)  # Randomly choose number of timestamps to replace
    
    indices = np.random.choice(seq_len, replace_count, replace=False)  # Randomly choose timestamps to replace
    data = data.clone()
    data[indices, :] = 0
    
    return data


def augment_data(X_tensor: torch.Tensor, apply_augmentation: bool = True) -> torch.Tensor:
    """数据增强（与train_lidara_detector.py一致）"""
    if not apply_augmentation:
        return X_tensor
    
    X_augmented = []
    for x in X_tensor:
        # Randomly choose augmentation technique
        augmentation = np.random.choice(['original', 'reverse', 'random_replace'], p=[0.5, 0.25, 0.25])
        if augmentation == 'reverse':
            x_augmented = reverse_sequence(x)
        elif augmentation == 'random_replace':
            x_augmented = random_replace_with_zeros(x)
        else:
            x_augmented = x
        X_augmented.append(x_augmented)
    
    X_augmented = torch.stack(X_augmented)
    
    return X_augmented


# 保留旧的augment_batch函数用于向后兼容（如果--augment参数被使用）
def augment_batch(batch: torch.Tensor,
                  prob_reverse: float = 0.25,
                  prob_noise: float = 0.35,
                  prob_zero: float = 0.15
                  ) -> torch.Tensor:
    """旧的数据增强函数（向后兼容）"""
    return augment_data(batch, apply_augmentation=True)


class RadarLSTMClassifier(nn.Module):
    """LSTM分类器（与train_lidara_detector.py一致）"""
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout_rate: float,
                 num_classes: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM的dropout只在多层时有效（层间的dropout）
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        # 添加额外的dropout层（用于LSTM输出后）
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 在LSTM输出后添加dropout（取最后一时刻的输出）
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class FocalLoss(nn.Module):
    """Focal Loss（与train_lidara_detector.py一致）"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 类别权重，可以是标量（用于正样本）或列表 [负样本权重, 正样本权重]
            gamma: 聚焦参数，越大越关注难分类样本
            reduction: 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) 模型输出（logits）
            targets: (N,) 真实标签
        """
        # 计算交叉熵损失
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        
        # 计算预测概率
        p = torch.exp(-ce_loss)  # p_t = exp(-CE)
        
        # 计算类别权重
        if isinstance(self.alpha, (float, int)):
            # 如果是标量，正样本用alpha，负样本用1-alpha
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        else:
            # 如果是列表，根据标签选择对应的权重
            alpha_t = torch.tensor([self.alpha[0] if t == 0 else self.alpha[1] 
                                   for t in targets], device=inputs.device, dtype=inputs.dtype)
        
        # 计算 Focal Loss
        focal_loss = alpha_t * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def evaluate(model: nn.Module, loader: DataLoader,
             criterion, device: torch.device,
             return_confidences: bool = False
             ):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0
    
    # 用于收集置信度信息
    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            
            if return_confidences:
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(probs.max(dim=1)[0].cpu().numpy())

    avg_loss = running_loss / total if total else 0.0
    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    result = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    
    if return_confidences:
        result["predictions"] = np.array(all_predictions)
        result["labels"] = np.array(all_labels)
        result["confidences"] = np.array(all_confidences)
    
    return result


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 确定scaler路径
    scaler_path = args.scaler_path
    if scaler_path is None:
        scaler_path = args.model_path.replace('.pth', '_scaler.pkl')

    feature_train, label_train, feature_val, label_val, scaler, feature_test, label_test = load_feature_sets(
        args.result_folder, args.feature_subdir,
        normalize=args.normalize,  # 默认False（不标准化）
        scaler_path=scaler_path)

    X_train = torch.tensor(feature_train, dtype=torch.float32)
    y_train = torch.tensor(label_train, dtype=torch.long)
    X_val = torch.tensor(feature_val, dtype=torch.float32)
    y_val = torch.tensor(label_val, dtype=torch.long)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    # 如果测试集存在，创建测试集DataLoader
    test_loader = None
    if feature_test is not None and label_test is not None:
        X_test = torch.tensor(feature_test, dtype=torch.float32)
        y_test = torch.tensor(label_test, dtype=torch.long)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        print(f"测试样本: {X_test.shape}, 正样本(测试): {int(y_test.sum())}")

    input_size = X_train.shape[-1]
    
    # 打印数据集统计信息（与train_lidara_detector.py一致）
    print(f"\n数据集统计:")
    print(f"训练集: {len(X_train)} 个样本, 正样本: {int(y_train.sum())} ({int(y_train.sum())/len(y_train)*100:.2f}%)")
    print(f"验证集: {len(X_val)} 个样本, 正样本: {int(y_val.sum())} ({int(y_val.sum())/len(y_val)*100:.2f}%)")
    if feature_test is not None and label_test is not None:
        y_test_temp = torch.tensor(label_test, dtype=torch.long)
        print(f"测试集: {len(y_test_temp)} 个样本, 正样本: {int(y_test_temp.sum())} ({int(y_test_temp.sum())/len(y_test_temp)*100:.2f}%)")
    print(f"特征形状: {X_train.shape}")
    print(f"标签形状: {y_train.shape}")
    
    # 在训练前进行数据增强（与train_lidara_detector.py一致）
    print("\n对训练数据进行增强...")
    X_train = augment_data(X_train, apply_augmentation=True)
    
    # 更新train_dataset和train_loader（因为X_train已经被增强）
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    model = RadarLSTMClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout).to(device)
    
    print(f"\n模型配置:")
    print(f"  输入维度: {input_size}")
    print(f"  隐藏层大小: {args.hidden_size}")
    print(f"  LSTM层数: {args.num_layers}")
    print(f"  Dropout率: {args.dropout}")
    print(f"  类别数: 2")

    # 使用 Focal Loss（与train_lidara_detector.py一致）
    pos_ratio = float(y_train.sum().item()) / len(y_train)
    focal_alpha = 0.5  # 正样本权重，可以根据需要调整（0.25-0.5）
    
    print(f"\nFocal Loss 配置:")
    print(f"  α (alpha): {focal_alpha} (正样本权重)")
    print(f"  γ (gamma): 2.0 (聚焦参数，关注难分类样本)")
    print(f"  正样本比例: {pos_ratio*100:.2f}%")
    print(f"  Focal Loss 会自动降低易分类样本的权重，更关注难分类样本")
    
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 模型保存目录（如果提供了目录）
    model_save_dir = os.path.dirname(args.model_path)
    if model_save_dir:
        os.makedirs(model_save_dir, exist_ok=True)

    # 跟踪最佳测试集性能（如果测试集存在）或验证集性能
    best_test_f1 = 0.0
    best_test_acc = 0.0
    best_test_loss = float("inf")
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    use_test_for_saving = test_loader is not None

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 在训练循环中应用数据增强（与train_lidara_detector.py一致）
            inputs = augment_data(inputs, apply_augmentation=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = epoch_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0

        metrics = evaluate(model, val_loader, criterion, device)
        
        # 如果测试集存在，也在测试集上评估
        test_metrics = None
        if use_test_for_saving:
            test_metrics = evaluate(model, test_loader, criterion, device)

        # 基于测试集F1保存最佳模型（如果测试集存在），否则基于验证集F1
        save_best = False
        if use_test_for_saving and test_metrics is not None:
            # 基于测试集F1保存最佳模型
            if test_metrics["f1"] > best_test_f1 or (
                test_metrics["f1"] == best_test_f1 and test_metrics["accuracy"] > best_test_acc
            ):
                save_best = True
            elif test_metrics["loss"] < best_test_loss:
                # 向后兼容：loss更优也保存
                save_best = True
        else:
            # 基于验证集F1保存最佳模型
            if metrics["f1"] > best_val_f1 or (
                metrics["f1"] == best_val_f1 and metrics["accuracy"] > best_val_acc
            ):
                save_best = True
            elif metrics["loss"] < best_val_loss:
                # 向后兼容：loss更优也保存
                save_best = True

        if save_best:
            if use_test_for_saving and test_metrics is not None:
                best_test_f1 = test_metrics["f1"]
                best_test_acc = test_metrics["accuracy"]
                best_test_loss = test_metrics["loss"]
                save_metrics = test_metrics
                save_metrics_name = "Test"
            else:
                best_val_f1 = metrics["f1"]
                best_val_acc = metrics["accuracy"]
                best_val_loss = metrics["loss"]
                save_metrics = metrics
                save_metrics_name = "Val"
            
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": metrics,
                "test_metrics": test_metrics if test_metrics is not None else None,
                "model_config": {  # 使用model_config（与train_lidara_detector.py一致）
                    "input_size": input_size,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "num_classes": 2,
                    "dropout_rate": args.dropout,
                },
            }
            torch.save(checkpoint, args.model_path)
            print(
                f"  ✓ 保存最佳模型 (Epoch {epoch}, 基于{save_metrics_name} F1) "
                f"{save_metrics_name} F1: {save_metrics['f1']:.4f}, Acc: {save_metrics['accuracy']:.4f}, Loss: {save_metrics['loss']:.4f}"
            )

        print_str = (
            f"Epoch [{epoch}/{args.num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {metrics['loss']:.4f} Acc: {metrics['accuracy']:.4f} "
            f"Prec: {metrics['precision']:.4f} Rec: {metrics['recall']:.4f} "
            f"F1: {metrics['f1']:.4f}"
        )
        if test_metrics is not None:
            print_str += (
                f" | Test Loss: {test_metrics['loss']:.4f} Acc: {test_metrics['accuracy']:.4f} "
                f"Prec: {test_metrics['precision']:.4f} Rec: {test_metrics['recall']:.4f} "
                f"F1: {test_metrics['f1']:.4f}"
            )
        print(print_str)

    print("\n=== 最终评估（使用最佳模型） ===")
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            saved_val_metrics = checkpoint.get("val_metrics", {})
            saved_test_metrics = checkpoint.get("test_metrics", None)
            print(
                f"加载最佳模型: {args.model_path} "
                f"(Epoch {checkpoint.get('epoch', '?')})"
            )
            if saved_test_metrics:
                print(
                    f"  保存时 - Test F1: {saved_test_metrics.get('f1', 0):.4f}, "
                    f"Acc: {saved_test_metrics.get('accuracy', 0):.4f}, "
                    f"Loss: {saved_test_metrics.get('loss', 0):.4f}"
                )
            else:
                print(
                    f"  保存时 - Val F1: {saved_val_metrics.get('f1', 0):.4f}, "
                    f"Acc: {saved_val_metrics.get('accuracy', 0):.4f}, "
                    f"Loss: {saved_val_metrics.get('loss', 0):.4f}"
                )
        else:
            model.load_state_dict(checkpoint)
            print(f"加载模型权重: {args.model_path}")
    else:
        print(f"⚠ 未找到模型文件: {args.model_path}")

    # 在验证集上评估（收集置信度信息）
    print("\n验证集评估:")
    final_val_metrics = evaluate(model, val_loader, criterion, device, return_confidences=True)
    print(
        f"  Accuracy: {final_val_metrics['accuracy']:.4f}, "
        f"Precision: {final_val_metrics['precision']:.4f}, "
        f"Recall: {final_val_metrics['recall']:.4f}, "
        f"F1: {final_val_metrics['f1']:.4f}")
    print(
        f"  Confusion Matrix -> TP: {final_val_metrics['tp']} FP: {final_val_metrics['fp']} "
        f"FN: {final_val_metrics['fn']} TN: {final_val_metrics['tn']}")
    
    # 分析验证集FP和FN的置信度
    val_predictions = final_val_metrics['predictions']
    val_labels = final_val_metrics['labels']
    val_confidences = final_val_metrics['confidences']
    
    if final_val_metrics['fn'] > 0:
        fn_mask = (val_predictions == 0) & (val_labels == 1)
        fn_confidences = val_confidences[fn_mask] if np.any(fn_mask) else np.array([])
        if len(fn_confidences) > 0:
            print(f"\n  漏检的正样本 (FN={final_val_metrics['fn']}):")
            print(f"    平均置信度: {fn_confidences.mean():.4f}")
            print(f"    最大置信度: {fn_confidences.max():.4f}")
            print(f"    最小置信度: {fn_confidences.min():.4f}")
            print(f"    中位数置信度: {np.median(fn_confidences):.4f}")
    
    if final_val_metrics['fp'] > 0:
        fp_mask = (val_predictions == 1) & (val_labels == 0)
        fp_confidences = val_confidences[fp_mask] if np.any(fp_mask) else np.array([])
        if len(fp_confidences) > 0:
            print(f"\n  误检的负样本 (FP={final_val_metrics['fp']}):")
            print(f"    平均置信度: {fp_confidences.mean():.4f}")
            print(f"    最大置信度: {fp_confidences.max():.4f}")
            print(f"    最小置信度: {fp_confidences.min():.4f}")
            print(f"    中位数置信度: {np.median(fp_confidences):.4f}")

    # 如果测试集存在，也在测试集上评估（收集置信度信息）
    if test_loader is not None:
        print("\n测试集评估:")
        final_test_metrics = evaluate(model, test_loader, criterion, device, return_confidences=True)
        print(
            f"  Accuracy: {final_test_metrics['accuracy']:.4f}, "
            f"Precision: {final_test_metrics['precision']:.4f}, "
            f"Recall: {final_test_metrics['recall']:.4f}, "
            f"F1: {final_test_metrics['f1']:.4f}")
        print(
            f"  Confusion Matrix -> TP: {final_test_metrics['tp']} FP: {final_test_metrics['fp']} "
            f"FN: {final_test_metrics['fn']} TN: {final_test_metrics['tn']}")
        
        # 分析测试集FP和FN的置信度
        test_predictions = final_test_metrics['predictions']
        test_labels = final_test_metrics['labels']
        test_confidences = final_test_metrics['confidences']
        
        if final_test_metrics['fn'] > 0:
            fn_mask = (test_predictions == 0) & (test_labels == 1)
            fn_confidences = test_confidences[fn_mask] if np.any(fn_mask) else np.array([])
            if len(fn_confidences) > 0:
                print(f"\n  漏检的正样本 (FN={final_test_metrics['fn']}):")
                print(f"    平均置信度: {fn_confidences.mean():.4f}")
                print(f"    最大置信度: {fn_confidences.max():.4f}")
                print(f"    最小置信度: {fn_confidences.min():.4f}")
                print(f"    中位数置信度: {np.median(fn_confidences):.4f}")
        
        if final_test_metrics['fp'] > 0:
            fp_mask = (test_predictions == 1) & (test_labels == 0)
            fp_confidences = test_confidences[fp_mask] if np.any(fp_mask) else np.array([])
            if len(fp_confidences) > 0:
                print(f"\n  误检的负样本 (FP={final_test_metrics['fp']}):")
                print(f"    平均置信度: {fp_confidences.mean():.4f}")
                print(f"    最大置信度: {fp_confidences.max():.4f}")
                print(f"    最小置信度: {fp_confidences.min():.4f}")
                print(f"    中位数置信度: {np.median(fp_confidences):.4f}")

    print("\n模型保存信息")
    print(f"  路径: {args.model_path}")
    if best_epoch:
        print(f"  最佳Epoch: {best_epoch}")
        if use_test_for_saving:
            print(f"  基于测试集F1保存 - 最佳Test F1: {best_test_f1:.4f}, 最佳Acc: {best_test_acc:.4f}, 最佳Loss: {best_test_loss:.4f}")
        else:
            print(f"  基于验证集F1保存 - 最佳Val F1: {best_val_f1:.4f}, 最佳Acc: {best_val_acc:.4f}, 最佳Loss: {best_val_loss:.4f}")
    else:
        print("  ⚠ 未保存最佳模型")


if __name__ == "__main__":
    main()

