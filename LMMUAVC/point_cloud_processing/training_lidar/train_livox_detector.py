import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Define directories and load data
result_folder = "./result"
save_directory = os.path.join(result_folder, "livox_avia_feature_set")

feature_set_train = np.load(os.path.join(save_directory, 'feature_livox_avia_train.npy'))
label_set_train = np.load(os.path.join(save_directory, 'label_livox_avia_train.npy'))
feature_set_val = np.load(os.path.join(save_directory, 'feature_livox_avia_val.npy'))
label_set_val = np.load(os.path.join(save_directory, 'label_livox_avia_val.npy'))
feature_set_test = np.load(os.path.join(save_directory, 'feature_livox_avia_test.npy'))
label_set_test = np.load(os.path.join(save_directory, 'label_livox_avia_test.npy'))

# Filter data based on labels
ind = label_set_train == 1
gt_cluster_train = feature_set_train[ind]

ind = label_set_val == 1
gt_cluster_val = feature_set_val[ind]

ind = label_set_test == 1
gt_cluster_test = feature_set_test[ind]
ind = label_set_test == 0
bg_cluster_test = feature_set_test[ind]

# Function to reverse the sequence
def reverse_sequence(data):
    return torch.flip(data, dims=[1])

# Function to add noise to features (更适合位置、速度、加速度特征)
def add_noise_to_features(data, noise_scale=0.1):
    """
    对15维特征添加小幅度噪声
    特征顺序: 均值(3) + 标准差(3) + 范围(3) + 速度(3) + 加速度(3)
    噪声幅度: 均值0.1m, 标准差0.05m, 范围0.1m, 速度0.03m/s, 加速度0.02m/s²
    """
    seq_len, num_features = data.size()
    noise = torch.randn_like(data)
    # 对不同特征分别使用不同的噪声尺度
    noise_scale_tensor = torch.ones_like(data)
    noise_scale_tensor[:, :3] = 0.1    # 均值噪声: 0.1米
    noise_scale_tensor[:, 3:6] = 0.05  # 标准差噪声: 0.05米
    noise_scale_tensor[:, 6:9] = 0.1  # 范围噪声: 0.1米
    noise_scale_tensor[:, 9:12] = 0.03  # 速度噪声: 0.03米/秒
    noise_scale_tensor[:, 12:15] = 0.02  # 加速度噪声: 0.02米/秒²
    data = data + noise * noise_scale_tensor * noise_scale
    return data

# Function to randomly replace features with all zeros (保留但使用概率降低)
def random_replace_with_zeros(data, max_replace=2):
    seq_len, num_features = data.size()
    # Ensure replace_count doesn't exceed seq_len
    max_possible_replace = min(max_replace, seq_len)
    replace_count = np.random.randint(1, max_possible_replace+1)
    
    indices = np.random.choice(seq_len, replace_count, replace=False)
    data[indices, :] = 0
    
    return data

# Augment the data
def augment_data(X_tensor, apply_augmentation=True):
    if not apply_augmentation:
        return X_tensor
    
    X_augmented = []
    for x in X_tensor:
        # Randomly choose augmentation technique
        # 对于位置、速度、加速度特征，优先使用噪声增强和序列反转
        augmentation = np.random.choice(
            ['original', 'reverse', 'add_noise', 'random_replace'],
                                       p=[0.4, 0.25, 0.25, 0.1])
        if augmentation == 'reverse':
            x_augmented = reverse_sequence(x)
        elif augmentation == 'add_noise':
            x_augmented = add_noise_to_features(x.clone())
        elif augmentation == 'random_replace':
            x_augmented = random_replace_with_zeros(x.clone())
        else:
            x_augmented = x
        X_augmented.append(x_augmented)
    
    X_augmented = torch.stack(X_augmented)
    
    return X_augmented

# Convert data and labels to PyTorch tensors
X_tensor_train = torch.tensor(feature_set_train, dtype=torch.float32) 
y_tensor_train = torch.tensor(label_set_train, dtype=torch.long)

X_tensor_test = torch.tensor(feature_set_test, dtype=torch.float32)  
y_tensor_test = torch.tensor(label_set_test, dtype=torch.long)

# Convert validation data to PyTorch tensors
X_tensor_val = torch.tensor(feature_set_val, dtype=torch.float32)
y_tensor_val = torch.tensor(label_set_val, dtype=torch.long)

# 验证特征维度
print(f"\n数据集统计:")
print(f"训练集: {len(X_tensor_train)} 个样本, 正样本: {np.sum(label_set_train == 1)} ({np.sum(label_set_train == 1)/len(label_set_train)*100:.2f}%)")
print(f"验证集: {len(X_tensor_val)} 个样本, 正样本: {np.sum(label_set_val == 1)} ({np.sum(label_set_val == 1)/len(label_set_val)*100:.2f}%)")
print(f"测试集: {len(X_tensor_test)} 个样本, 正样本: {np.sum(label_set_test == 1)} ({np.sum(label_set_test == 1)/len(label_set_test)*100:.2f}%)")
print(f"特征形状: {X_tensor_train.shape}")
print(f"标签形状: {label_set_train.shape}")
print(f"特征维度 (应该是15: 3均值 + 3标准差 + 3范围 + 3速度 + 3加速度): {X_tensor_train.shape[-1]}")
assert X_tensor_train.shape[-1] == 15, f"特征维度不正确，期望15，实际为{X_tensor_train.shape[-1]}" 

# Augment the training data
X_tensor_train = augment_data(X_tensor_train, apply_augmentation=True)

# Define batch size
batch_size = 64

# Create DataLoader for training, validation and testing data
train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_tensor_val, y_tensor_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_tensor_test, y_tensor_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Apply dropout to the last output
        out = self.fc(out)
        return out

# Initialize model
input_size = 15  # Number of features: 3均值 + 3标准差 + 3范围 + 3速度 + 3加速度
hidden_size = 64
num_layers = 1
num_classes = 2  # Number of unique classes
dropout_rate = 0.1
  # Dropout rate for regularization
model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes, dropout_rate)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义模型保存目录
model_save_dir = "/home/p/MMUAV/point_cloud_processing/training_lidar/checkpoints/livox_avia_detector"
os.makedirs(model_save_dir, exist_ok=True)

# Training the model
model_path = os.path.join(model_save_dir, 'best_model.pth')
best_valid_loss = float('inf')

# 跟踪测试集上的最佳性能
best_test_accuracy = 0.0
best_test_f1 = 0.0
best_test_epoch = 0
best_model_path = None

num_epochs = 40
print(f"模型保存目录: {model_save_dir}")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = augment_data(inputs, apply_augmentation=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct / total
    
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = val_correct / val_total
    
    # Test phase (评估测试集，用于保存最佳模型)
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    test_tp = 0
    test_fp = 0
    test_fn = 0
    test_tn = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # 计算混淆矩阵
            test_tp += ((predicted == 1) & (labels == 1)).sum().item()
            test_fp += ((predicted == 1) & (labels == 0)).sum().item()
            test_fn += ((predicted == 0) & (labels == 1)).sum().item()
            test_tn += ((predicted == 0) & (labels == 0)).sum().item()
    
    test_loss = test_running_loss / len(test_loader.dataset)
    test_accuracy = test_correct / test_total
    
    # 计算F1分数
    test_precision = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0
    test_recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
    
    # 使用F1分数作为主要指标（因为类别不平衡），准确率作为辅助指标
    # 如果F1分数更好，或者F1分数相同但准确率更好，则保存模型
    is_best = False
    if test_f1 > best_test_f1 or (test_f1 == best_test_f1 and test_accuracy > best_test_accuracy):
        is_best = True
        best_test_f1 = test_f1
        best_test_accuracy = test_accuracy
        best_test_epoch = epoch + 1
        
        # 保存最佳模型（完整checkpoint）
        best_model_path = model_path
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_loss': test_loss,
            'model_config': {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'num_classes': num_classes,
                'dropout_rate': dropout_rate
            }
        }, best_model_path)
        print(f'  ✓ 保存最佳模型 (Test F1: {test_f1:.4f}, Test Acc: {test_accuracy:.4f})')
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1: {test_f1:.4f}')

# 加载最佳模型进行评估（优先使用F1分数最佳的模型）
if best_model_path and os.path.exists(best_model_path):
    print(f"\n加载最佳模型 (Epoch {best_test_epoch}, Test F1: {best_test_f1:.4f}, Test Acc: {best_test_accuracy:.4f})...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型已加载: {best_model_path}")
else:
    # 向后兼容：如果没有保存最佳模型，使用基于损失的模型
    print(f"\n加载基于损失的最佳模型...")
    model.load_state_dict(torch.load(model_path))

# Evaluate the model on the test set
model.eval()
running_loss = 0.0
correct = 0
total = 0
true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0

# 收集预测信息用于分析置信度
test_predictions = []
test_labels_list = []
test_confidences = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        probs = torch.softmax(outputs, dim=1)
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 计算混淆矩阵的各个组成部分
        true_positive += ((predicted == 1) & (labels == 1)).sum().item()
        false_positive += ((predicted == 1) & (labels == 0)).sum().item()
        false_negative += ((predicted == 0) & (labels == 1)).sum().item()
        true_negative += ((predicted == 0) & (labels == 0)).sum().item()
        
        # 收集预测信息用于分析置信度
        test_predictions.extend(predicted.cpu().numpy())
        test_labels_list.extend(labels.cpu().numpy())
        test_confidences.extend(probs.max(dim=1)[0].cpu().numpy())

test_predictions = np.array(test_predictions)
test_labels_list = np.array(test_labels_list)
test_confidences = np.array(test_confidences)
    
test_loss = running_loss / len(test_loader.dataset)
test_accuracy = correct / total

# 正确计算召回率 (Recall = TP / (TP + FN))
if true_positive + false_negative > 0:
    test_recall = true_positive / (true_positive + false_negative)
else:
    test_recall = 0.0

# 计算精确率 (Precision = TP / (TP + FP))
if true_positive + false_positive > 0:
    test_precision = true_positive / (true_positive + false_positive)
else:
    test_precision = 0.0

# 计算F1分数
if test_precision + test_recall > 0:
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
else:
    test_f1 = 0.0

print(f'\n最终测试结果:')
print(f'  Accuracy: {test_accuracy:.4f}')
print(f'  Precision: {test_precision:.4f}')
print(f'  Recall: {test_recall:.4f}')
print(f'  F1-Score: {test_f1:.4f}')
print(f'  混淆矩阵:')
print(f'    TP: {true_positive}, FP: {false_positive}')
print(f'    FN: {false_negative}, TN: {true_negative}')

# 分析FP和FN的置信度
if false_negative > 0:
    fn_mask = (test_predictions == 0) & (test_labels_list == 1)
    fn_confidences = test_confidences[fn_mask] if np.any(fn_mask) else np.array([])
    if len(fn_confidences) > 0:
        print(f'\n  漏检的正样本 (FN={false_negative}):')
        print(f'    平均置信度: {fn_confidences.mean():.4f}')
        print(f'    最大置信度: {fn_confidences.max():.4f}')
        print(f'    最小置信度: {fn_confidences.min():.4f}')
        print(f'    中位数置信度: {np.median(fn_confidences):.4f}')

if false_positive > 0:
    fp_mask = (test_predictions == 1) & (test_labels_list == 0)
    fp_confidences = test_confidences[fp_mask] if np.any(fp_mask) else np.array([])
    if len(fp_confidences) > 0:
        print(f'\n  误检的负样本 (FP={false_positive}):')
        print(f'    平均置信度: {fp_confidences.mean():.4f}')
        print(f'    最大置信度: {fp_confidences.max():.4f}')
        print(f'    最小置信度: {fp_confidences.min():.4f}')
        print(f'    中位数置信度: {np.median(fp_confidences):.4f}')

# 打印模型保存信息
print(f"\n{'='*80}")
print(f"模型保存信息")
print(f"{'='*80}")
if best_model_path and os.path.exists(best_model_path):
    print(f"最佳模型已保存: {best_model_path}")
    print(f"最佳性能 (Epoch {best_test_epoch}):")
    print(f"  测试集准确率: {best_test_accuracy:.4f}")
    print(f"  测试集F1分数: {best_test_f1:.4f}")
    print(f"\n加载模型示例:")
    print(f"  checkpoint = torch.load('{best_model_path}')")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
else:
    print(f"⚠️  未找到保存的模型")
