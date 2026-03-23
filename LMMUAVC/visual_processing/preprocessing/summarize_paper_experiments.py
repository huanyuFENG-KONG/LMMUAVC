#!/usr/bin/env python3
"""
汇总论文实验结果脚本

从所有实验检查点中提取性能指标，生成与论文表格一致的汇总文件
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="汇总论文实验结果")
    parser.add_argument("--experiments-dir", type=str, default="./experiments_paper",
                        help="实验目录")
    parser.add_argument("--output-file", type=str, default="./experiments_paper/paper_results_summary.md",
                        help="输出文件路径")
    return parser.parse_args()


def extract_metrics_from_checkpoint(checkpoint_path):
    """
    从检查点文件中提取性能指标
    
    Returns:
        dict with keys: accuracy, precision, recall, f1, loss, epoch
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        metrics = {}
        
        # 尝试从不同位置提取指标
        if 'val_metrics' in checkpoint:
            val_metrics = checkpoint['val_metrics']
            metrics['accuracy'] = val_metrics.get('accuracy', 0.0)
            metrics['precision'] = val_metrics.get('precision', 0.0)
            metrics['recall'] = val_metrics.get('recall', 0.0)
            metrics['f1'] = val_metrics.get('f1', 0.0)
            metrics['loss'] = val_metrics.get('loss', float('inf'))
        
        if 'test_metrics' in checkpoint and checkpoint['test_metrics'] is not None:
            test_metrics = checkpoint['test_metrics']
            metrics['test_accuracy'] = test_metrics.get('accuracy', 0.0)
            metrics['test_f1'] = test_metrics.get('f1', 0.0)
        
        metrics['epoch'] = checkpoint.get('epoch', 0)
        
        # 提取模型配置
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            metrics['model_type'] = model_config.get('model_type', 'unknown')
            metrics['hidden_dim'] = model_config.get('hidden_dim', 0)
        
        return metrics
    except Exception as e:
        print(f"警告: 无法加载 {checkpoint_path}: {e}")
        return None


def extract_metrics_from_log(log_path):
    """
    从训练日志中提取最终性能指标
    
    Returns:
        dict with metrics
    """
    metrics = {}
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # 查找最后一行的验证指标
        for line in reversed(lines):
            if 'Val' in line and ('Acc:' in line or 'F1:' in line):
                # 提取数字
                acc_match = re.search(r'Acc:\s*([\d.]+)', line)
                f1_match = re.search(r'F1:\s*([\d.]+)', line)
                
                if acc_match:
                    metrics['accuracy'] = float(acc_match.group(1))
                if f1_match:
                    metrics['f1'] = float(f1_match.group(1))
                break
    except Exception as e:
        print(f"警告: 无法解析日志 {log_path}: {e}")
    
    return metrics


def count_parameters(checkpoint_path):
    """计算模型参数量"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        total_params = sum(p.numel() for p in state_dict.values())
        return total_params
    except:
        return None


def generate_summary(experiments_dir, output_file):
    """生成实验结果汇总"""
    experiments_dir = Path(experiments_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = defaultdict(dict)
    
    # 扫描所有检查点目录
    checkpoints_dir = experiments_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"错误: 检查点目录不存在: {checkpoints_dir}")
        return
    
    for exp_dir in checkpoints_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        checkpoint_path = exp_dir / "best_model.pth"
        if not checkpoint_path.exists():
            continue
        
        exp_name = exp_dir.name
        metrics = extract_metrics_from_checkpoint(checkpoint_path)
        if metrics:
            results[exp_name] = metrics
            results[exp_name]['params'] = count_parameters(checkpoint_path)
    
    # 生成 Markdown 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 论文实验结果汇总\n\n")
        f.write("本文档汇总了所有论文实验的结果。\n\n")
        
        # 4.3.1 单模态基线
        f.write("## 4.3.1 单模态基线\n\n")
        f.write("| Modality | Validation Accuracy | Notes |\n")
        f.write("|----------|-------------------|-------|\n")
        for modality in ['image', 'livox', 'lidar360', 'radar']:
            key = f'single_modality_{modality}'
            if key in results:
                acc = results[key].get('accuracy', 0.0) * 100
                f.write(f"| {modality.capitalize()} only | {acc:.1f}% | - |\n")
        f.write("\n")
        
        # 4.3.2 简单融合基线
        f.write("## 4.3.2 简单融合基线\n\n")
        f.write("| Method | Validation Accuracy | Overfitting Gap | Parameters |\n")
        f.write("|--------|-------------------|-----------------|------------|\n")
        
        baseline_experiments = {
            'baseline_concatenation': 'Concatenation',
            'baseline_attention_fusion': 'Attention Fusion',
            'ablation_full': 'Ours (Compact Transformer)',
        }
        
        for exp_key, method_name in baseline_experiments.items():
            if exp_key in results:
                acc = results[exp_key].get('accuracy', 0.0) * 100
                # 计算过拟合差距（需要训练准确率，暂时用验证损失估算）
                gap = "N/A"
                params = results[exp_key].get('params', 0)
                if params:
                    params_str = f"{params/1e6:.1f}M"
                else:
                    params_str = "N/A"
                f.write(f"| {method_name} | {acc:.1f}% | {gap} | {params_str} |\n")
        f.write("\n")
        
        # 4.4.1 缺失数据建模消融
        f.write("## 4.4.1 缺失数据建模消融\n\n")
        f.write("| Component | Validation Accuracy | F1-Score |\n")
        f.write("|-----------|-------------------|----------|\n")
        
        ablation_experiments = {
            'ablation_no_missing_handling': 'No missing handling (zero-padding)',
            'ablation_mask_only': 'Missing mask only (no learnable embeddings)',
            'ablation_embeddings_only': 'Learnable embeddings only (no mask)',
            'ablation_full': 'Learnable embeddings + mask',
        }
        
        for exp_key, component_name in ablation_experiments.items():
            if exp_key in results:
                acc = results[exp_key].get('accuracy', 0.0) * 100
                f1 = results[exp_key].get('f1', 0.0)
                f.write(f"| {component_name} | {acc:.1f}% | {f1:.3f} |\n")
        f.write("\n")
        
        # 4.4.2 权重机制消融
        f.write("## 4.4.2 权重机制消融\n\n")
        f.write("| Weighting | Validation Accuracy | Notes |\n")
        f.write("|-----------|-------------------|-------|\n")
        
        weighting_experiments = {
            'ablation_no_weighting': 'No weighting (equal weights)',
            'ablation_confidence_only': 'Confidence weighting only',
            'ablation_density_only': 'Density weighting only',
            'ablation_both_weighting': 'Both (confidence + density)',
        }
        
        baseline_acc = None
        for exp_key, weighting_name in weighting_experiments.items():
            if exp_key in results:
                acc = results[exp_key].get('accuracy', 0.0) * 100
                if baseline_acc is None and 'no_weighting' in exp_key:
                    baseline_acc = acc
                
                if baseline_acc:
                    improvement = acc - baseline_acc
                    notes = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
                else:
                    notes = "-"
                
                f.write(f"| {weighting_name} | {acc:.1f}% | {notes} |\n")
        f.write("\n")
        
        # 4.4.3 架构变体消融
        f.write("## 4.4.3 架构变体消融\n\n")
        f.write("| Architecture | Parameters | Validation Accuracy | Overfitting Gap |\n")
        f.write("|--------------|------------|-------------------|-----------------|\n")
        
        architecture_experiments = {
            'architecture_tiny_mlp': 'Tiny MLP',
            'architecture_compact_transformer': 'Compact Transformer',
            'architecture_efficient_fusion': 'Efficient Fusion',
        }
        
        for exp_key, arch_name in architecture_experiments.items():
            if exp_key in results:
                params = results[exp_key].get('params', 0)
                acc = results[exp_key].get('accuracy', 0.0) * 100
                gap = "N/A"
                
                if params:
                    params_str = f"{params/1e6:.1f}M"
                else:
                    params_str = "N/A"
                
                f.write(f"| {arch_name} | {params_str} | {acc:.1f}% | {gap} |\n")
        f.write("\n")
        
        # 4.4.4 模态贡献分析
        f.write("## 4.4.4 模态贡献分析\n\n")
        f.write("| Modalities | Validation Accuracy | Relative Improvement |\n")
        f.write("|------------|-------------------|---------------------|\n")
        
        modality_experiments = {
            'modality_image': 'Image only',
            'modality_image_livox': 'Image + Livox',
            'modality_image_lidar360': 'Image + Lidar 360',
            'modality_image_radar': 'Image + Radar',
            'modality_image_livox_lidar360': 'Image + Livox + Lidar 360',
            'modality_image_livox_lidar360_radar': 'Image + Livox + Lidar 360 + Radar',
        }
        
        baseline_acc = None
        for exp_key, modality_combo in modality_experiments.items():
            if exp_key in results:
                acc = results[exp_key].get('accuracy', 0.0) * 100
                if baseline_acc is None and 'image' in exp_key and exp_key.count('_') == 1:
                    baseline_acc = acc
                
                if baseline_acc:
                    improvement = acc - baseline_acc
                    improvement_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
                else:
                    improvement_str = "Baseline"
                
                f.write(f"| {modality_combo} | {acc:.1f}% | {improvement_str} |\n")
        f.write("\n")
    
    print(f"实验结果汇总已保存到: {output_file}")


def main():
    args = parse_args()
    generate_summary(args.experiments_dir, args.output_file)


if __name__ == "__main__":
    main()

