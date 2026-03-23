"""
完整的特征提取和对齐流程。

该脚本会：
1. 运行数据对齐（preprocessing_improved.py）
2. 图像检测（detect_mmaud.py）- 检测无人机并裁剪图像
3. 点云检测（livox_avia_detector.py）- 检测点云中的目标并提取点云子集
4. 从检测结果提取图像特征（ConvNeXt）
5. 从检测结果提取点云特征（PointNeXt）
6. 准备训练所需的所有文件
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="完整的特征提取和对齐流程"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=["seq0001", "seq0002", "seq0003", "seq0004", "seq0005", "seq0006", "seq0007", "seq0008"],
        help="要处理的序列列表（默认: 全部序列）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./out",
        help="对齐结果输出目录",
    )
    parser.add_argument(
        "--point-cloud-dir",
        type=str,
        help="点云数据目录（如果使用目录模式）",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="图像数据目录（如果使用目录模式）",
    )
    parser.add_argument(
        "--pointnext-cfg",
        type=str,
        default="PointNeXt/cfgs/shapenetpart/pointnext-s.yaml",
        help="PointNeXt配置文件路径",
    )
    parser.add_argument(
        "--pointnext-pretrained",
        type=str,
        help="PointNeXt预训练模型路径",
    )
    parser.add_argument(
        "--pointnext-output",
        type=str,
        default="./features/pointnext_features.pt",
        help="PointNeXt特征输出路径（Livox Avia，如果不指定，会根据序列名称自动生成）",
    )
    parser.add_argument(
        "--lidar360-pointnext-output",
        type=str,
        default="./features/lidar360_pointnext_features.pt",
        help="Lidar 360 PointNeXt特征输出路径",
    )
    parser.add_argument(
        "--radar-pointnext-output",
        type=str,
        default="./features/radar_pointnext_features.pt",
        help="Radar Enhance PointNeXt特征输出路径",
    )
    parser.add_argument(
        "--convnext-output",
        type=str,
        default="./features/convnext_features.pt",
        help="ConvNeXt特征输出路径（如果不指定，会根据序列名称自动生成）",
    )
    parser.add_argument(
        "--per-sequence-output",
        action="store_true",
        help="为每个序列分别保存特征文件（文件名包含序列名称，便于后续合并）",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="跳过数据对齐步骤（如果已经对齐过）",
    )
    parser.add_argument(
        "--skip-pointnext",
        action="store_true",
        help="跳过点云特征提取（如果已经提取过）",
    )
    parser.add_argument(
        "--skip-convnext",
        action="store_true",
        help="跳过图像特征提取（如果已经提取过）",
    )
    parser.add_argument(
        "--alignment-strategy",
        type=str,
        default="time_window",
        choices=["nearest", "time_window", "one_to_many", "downsample"],
        help="对齐策略（用于preprocessing_improved.py的数据对齐阶段）",
    )
    parser.add_argument(
        "--feature-alignment-mode",
        type=str,
        default="multimodal",
        choices=["multimodal", "image_led", "pointcloud_led"],
        help="特征对齐模式：multimodal（无主模态，时间窗口内对齐，默认）、image_led（以图像为主导）、pointcloud_led（以点云为主导）",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=0.1,
        help="最大时间差（秒）",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=0.4,
        help="时间窗口大小（秒），用于图像采样和特征对齐（默认0.4秒）",
    )
    parser.add_argument(
        "--window-sizes",
        type=float,
        nargs="+",
        help="时间窗口大小列表（用于实验，会循环处理每个窗口大小）。如果指定此参数，将覆盖 --window-size",
    )
    parser.add_argument(
        "--experiment-mode",
        action="store_true",
        help="实验模式：当使用 --window-sizes 时，自动为每个窗口大小创建独立的特征目录（features_window_{size}）",
    )
    parser.add_argument(
        "--frames-per-window",
        type=int,
        default=8,
        help="每个时间窗口内选取的图像帧数（默认8帧）",
    )
    parser.add_argument(
        "--skip-image-detection",
        action="store_true",
        help="跳过图像检测步骤（如果已经检测过）",
    )
    parser.add_argument(
        "--skip-pointcloud-detection",
        action="store_true",
        help="跳过点云检测步骤（如果已经检测过）",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolo11s.pt",
        help="YOLO11模型权重路径",
    )
    parser.add_argument(
        "--yolo-source",
        type=str,
        help="YOLO检测的图像源目录（默认使用data-root下的Image目录）",
    )
    parser.add_argument(
        "--yolo-project",
        type=str,
        default="runs/detect",
        help="YOLO检测结果保存目录",
    )
    parser.add_argument(
        "--yolo-name",
        type=str,
        default="mmaud_exp",
        help="YOLO检测结果名称",
    )
    parser.add_argument(
        "--yolo-conf-thres",
        type=float,
        default=0.3,  # 降低默认阈值，提高无人机检测率（原来0.65可能过高，会漏检）
        help="YOLO置信度阈值（建议0.25-0.4，过低会有更多误检）",
    )
    parser.add_argument(
        "--yolo-half",
        action="store_true",
        default=True,
        help="启用FP16半精度推理（默认启用，加速约2倍）",
    )
    parser.add_argument(
        "--yolo-no-half",
        dest="yolo_half",
        action="store_false",
        help="禁用FP16半精度推理",
    )
    parser.add_argument(
        "--yolo-batch-size",
        type=int,
        default=16,
        help="YOLO批处理大小（默认16，避免OOM；大量图像时自动降低；RTX4090可尝试32）",
    )
    parser.add_argument(
        "--yolo-save-crop",
        action="store_true",
        default=True,
        help="保存裁剪的检测框",
    )
    parser.add_argument(
        "--livox-detector-model",
        type=str,
        help="Livox Avia检测器模型路径（如果跳过检测则不需要）",
    )
    parser.add_argument(
        "--livox-output-subdir",
        type=str,
        default="detections",
        help="点云检测结果保存子目录",
    )
    parser.add_argument(
        "--livox-metadata-filename",
        type=str,
        default="detections_metadata.csv",
        help="点云检测元数据文件名",
    )
    parser.add_argument(
        "--lidar360-detector-model",
        type=str,
        help="Lidar 360 检测器模型路径（可选）",
    )
    parser.add_argument(
        "--lidar360-output-subdir",
        type=str,
        default="lidar360_detections",
        help="Lidar 360 检测结果保存子目录",
    )
    parser.add_argument(
        "--lidar360-metadata-filename",
        type=str,
        default="lidar360_detections_metadata.csv",
        help="Lidar 360 检测元数据文件名",
    )
    parser.add_argument(
        "--radar-detector-model",
        type=str,
        help="Radar Enhance 检测器模型路径（可选）",
    )
    parser.add_argument(
        "--radar-output-subdir",
        type=str,
        default="radar_detections",
        help="Radar Enhance 检测结果保存子目录",
    )
    parser.add_argument(
        "--radar-metadata-filename",
        type=str,
        default="radar_detections_metadata.csv",
        help="Radar Enhance 检测元数据文件名",
    )
    parser.add_argument(
        "--radar-prob-threshold",
        type=float,
        default=0.001,  # 进一步降低阈值以获取更多检测结果
        help="Radar Enhance 检测概率阈值（默认0.001，如果scaler已加载可提高到0.3-0.5）",
    )
    parser.add_argument(
        "--detection-image-dir",
        type=str,
        help="检测后的裁剪图像目录（如果已检测，用于特征提取）",
    )
    parser.add_argument(
        "--detection-pointcloud-dir",
        type=str,
        help="检测后的点云目录（如果已检测，用于特征提取）",
    )
    parser.add_argument(
        "--detection-metadata-csv",
        type=str,
        help="点云检测元数据CSV文件路径（如果已检测）",
    )
    return parser.parse_args()


def run_alignment(args):
    """运行数据对齐。"""
    if args.skip_alignment:
        print("跳过数据对齐步骤")
        return
    
    print("=" * 60)
    print("步骤 1: 数据对齐")
    print("=" * 60)
    
    preprocessing_script = Path(__file__).parent / "preprocessing_improved.py"
    
    # 构建命令，传递参数给 preprocessing_improved.py
    cmd = [
        sys.executable,
        str(preprocessing_script),
        "--data-root", args.data_root,
        "--alignment-strategy", args.alignment_strategy,
        "--max-time-diff", str(args.max_time_diff),
    ]
    
    # 检测数据目录结构，确定prefix和输出目录
    data_root_path = Path(args.data_root)
    data_split = None  # 'train' 或 'val' 或 'test' 或 None
    prefix = None
    
    # 情况1: 数据在 data_root/train/ 或 data_root/val/ 或 data_root/test/ 下
    if (data_root_path / "train").exists() or (data_root_path / "val").exists() or (data_root_path / "test").exists():
        if (data_root_path / "train").exists():
            prefix = "train"
            data_split = "train"
        elif (data_root_path / "val").exists():
            prefix = "val"
            data_split = "val"
        elif (data_root_path / "test").exists():
            prefix = "test"
            data_split = "test"
    # 情况2: data_root 本身就是 train/val/test 目录（如 /home/p/MMUAV/test）
    elif data_root_path.name in ["train", "val", "test"]:
        data_split = data_root_path.name
        # 不需要设置 prefix，因为序列直接在 data_root 下
        # 需要调整 data_root 为父目录，并设置 prefix
        parent_root = data_root_path.parent
        prefix = data_root_path.name
        cmd[cmd.index("--data-root") + 1] = str(parent_root)
    # 情况3: 其他情况，默认为 train
    else:
        data_split = "train"
    
    if prefix:
        cmd.extend(["--prefix", prefix])
    
    # 根据指定的序列，设置seq-ids参数
    # 从序列名称提取数字（如 seq0002 -> 2）
    seq_numbers = []
    for seq_name in args.sequences:
        # 尝试从 seq0002 或 seq2 提取数字
        if seq_name.startswith("seq"):
            try:
                seq_num = int(seq_name[3:])  # 去掉 "seq" 前缀
                seq_numbers.append(seq_num)
            except ValueError:
                print(f"警告: 无法从序列名称 {seq_name} 提取数字，将跳过该序列的对齐")
    
    if seq_numbers:
        # 使用--seq-ids参数只处理指定的序列
        cmd.extend(["--seq-ids"] + [str(s) for s in seq_numbers])
    
    # 设置输出目录：确保时间线文件保存在 out/train/ 或 out/val/ 下
    if args.output_dir:
        # 如果指定了输出目录，创建对应的 train/val 子目录
        output_dir_path = Path(args.output_dir)
        output_dir_with_split = output_dir_path / data_split
        cmd.extend(["--output-dir", str(output_dir_with_split)])
    else:
        # 默认输出到 out/train/ 或 out/val/
        default_output = Path("out") / data_split
        cmd.extend(["--output-dir", str(default_output)])
    
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print("警告: 数据对齐步骤可能失败，请检查输出")
    else:
        print("数据对齐完成")


def sample_images_by_time_window(
    data_root: Path,
    sequences: list,
    window_size: float = 0.4,
    frames_per_window: int = 8
) -> Optional[Path]:
    """
    以图像为主，按时间窗口采样图像。
    
    新策略：
    1. 从所有图像中，按时间顺序，每window_size秒采样frames_per_window帧图像
    2. 对于每个时间窗口，检测这frames_per_window帧图像
    3. 在检测到无人机的窗口内，搜索是否有对应的点云检测结果
    
    Args:
        data_root: 数据根目录
        sequences: 序列名称列表
        window_size: 时间窗口大小（秒，默认0.4秒）
        frames_per_window: 每个时间窗口内的图像帧数（默认8帧）
    
    Returns:
        采样图像的临时目录路径，如果失败则返回None
    """
    # 创建采样图像的临时目录
    temp_image_dir = data_root / ".temp_sampled_images"
    if temp_image_dir.exists():
        shutil.rmtree(temp_image_dir)
    temp_image_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有序列的所有图像时间戳（从Image目录中读取所有图像文件）
    all_sequence_images = {}  # seq_name -> sorted list of (timestamp, file_path)
    
    print(f"正在从Image目录读取所有图像时间戳（时间窗口: {window_size}秒，每窗口{frames_per_window}帧）...")
    for seq_name in sequences:
        seq_image_dir = data_root / seq_name / "Image"
        if seq_image_dir.exists():
            image_files = list(seq_image_dir.glob("*.png"))
            image_timestamps = []
            for img_file in image_files:
                try:
                    img_ts = float(img_file.stem)
                    image_timestamps.append((img_ts, img_file))
                except ValueError:
                    continue
            if image_timestamps:
                image_timestamps.sort(key=lambda x: x[0])  # 按时间戳排序
                all_sequence_images[seq_name] = image_timestamps
                print(f"  序列 {seq_name}: 找到 {len(image_timestamps)} 个图像文件")
    
    if not all_sequence_images:
        print("警告: 未找到任何图像文件，无法进行采样")
        return None
    
    # 按时间窗口采样图像
    sampled_image_files = []  # 存储 (seq_name, timestamp, file_path)
    
    for seq_name, image_list in all_sequence_images.items():
        if not image_list:
            continue
        
        # 按时间窗口采样
        window_start = image_list[0][0]  # 第一个图像的时间戳
        window_images = []
        
        for img_ts, img_file in image_list:
            # 如果当前图像在窗口内
            if img_ts < window_start + window_size:
                window_images.append((img_ts, img_file))
            else:
                # 窗口已满，采样窗口内的图像
                if len(window_images) > 0:
                    # 如果窗口内图像数量超过frames_per_window，均匀采样
                    if len(window_images) > frames_per_window:
                        # 均匀采样frames_per_window帧
                        step = len(window_images) / frames_per_window
                        sampled_indices = [int(i * step) for i in range(frames_per_window)]
                        for idx in sampled_indices:
                            if idx < len(window_images):
                                ts, file = window_images[idx]
                                sampled_image_files.append((seq_name, ts, file))
                    else:
                        # 窗口内图像数量不足，全部采样
                        for ts, file in window_images:
                            sampled_image_files.append((seq_name, ts, file))
                
                # 开始新窗口
                window_start = img_ts
                window_images = [(img_ts, img_file)]
        
        # 处理最后一个窗口
        if len(window_images) > 0:
            if len(window_images) > frames_per_window:
                step = len(window_images) / frames_per_window
                sampled_indices = [int(i * step) for i in range(frames_per_window)]
                for idx in sampled_indices:
                    if idx < len(window_images):
                        ts, file = window_images[idx]
                        sampled_image_files.append((seq_name, ts, file))
            else:
                for ts, file in window_images:
                    sampled_image_files.append((seq_name, ts, file))
    
    print(f"按时间窗口采样了 {len(sampled_image_files)} 个图像（时间窗口: {window_size}秒，每窗口{frames_per_window}帧）")
    
    # 创建符号链接
    linked_count = 0
    for seq_name, img_ts, img_file in sampled_image_files:
        link_name = f"{seq_name}_{img_file.name}"
        link_path = temp_image_dir / link_name
        
        if not link_path.exists():
            try:
                link_path.symlink_to(img_file.resolve())
                linked_count += 1
            except OSError:
                # 如果符号链接失败，复制文件
                shutil.copy2(img_file, link_path)
                linked_count += 1
    
    print(f"已链接 {linked_count} 个采样图像到临时目录: {temp_image_dir}")
    
    if linked_count == 0:
        print("警告: 未找到任何采样图像")
        return None
    
    return temp_image_dir


def sample_images_by_livox_timeline(
    timeline_dir: Path,
    data_root: Path,
    sequences: list,
    image_ratio: int = 2,
    time_window: float = 0.1,
    image_search_window: float = 2.0  # 用于查找额外图像的搜索窗口（比对齐窗口更大，确保能找到足够的图像）
) -> Optional[Path]:
    """
    基于时间线文件，按照 livox:Image=1:2 的比例采样图像。
    
    Args:
        timeline_dir: 时间线文件目录（out/train 或 out/val）
        data_root: 数据根目录
        sequences: 序列名称列表
        image_ratio: 每个livox对应的图像数量（默认2，即1:2比例）
        time_window: 时间窗口大小（秒）
    
    Returns:
        采样图像的临时目录路径，如果失败则返回None
    """
    # 查找时间线文件
    timeline_files = sorted(timeline_dir.glob("*.csv"))
    if not timeline_files:
        print(f"警告: 未找到时间线文件: {timeline_dir}")
        return None
    
    # 读取所有时间线文件
    all_timelines = []
    for timeline_file in timeline_files:
        try:
            df = pd.read_csv(timeline_file, sep="\t", dtype={'seq': str})
            all_timelines.append(df)
        except Exception as e:
            print(f"警告: 读取时间线文件失败 {timeline_file}: {e}")
            continue
    
    if not all_timelines:
        print("错误: 无法读取任何时间线文件")
        return None
    
    timeline_df = pd.concat(all_timelines, ignore_index=True)
    
    # 创建采样图像的临时目录
    temp_image_dir = data_root / ".temp_sampled_images"
    if temp_image_dir.exists():
        shutil.rmtree(temp_image_dir)
    temp_image_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集要采样的图像时间戳
    sampled_image_timestamps = set()
    
    # 对于每个有livox_avia时间戳的行，采样对应的图像
    livox_col = 'livox_avia'
    image_col = 'Image'
    seq_col = 'seq'
    
    if livox_col not in timeline_df.columns or image_col not in timeline_df.columns:
        print(f"错误: 时间线文件缺少必要的列: {livox_col} 或 {image_col}")
        return None
    
    # 筛选出有livox和图像时间戳的行
    valid_rows = timeline_df[
        (timeline_df[livox_col].notna()) & 
        (timeline_df[livox_col].astype(str).str.strip() != '') &
        (timeline_df[image_col].notna()) &
        (timeline_df[image_col].astype(str).str.strip() != '')
    ].copy()
    
    print(f"时间线文件中共有 {len(valid_rows)} 行有效数据（有livox和图像时间戳）")
    
    # 按livox时间戳分组，每个livox采样image_ratio个图像
    # 将数据转换为浮点数以便比较
    valid_rows[livox_col] = pd.to_numeric(valid_rows[livox_col], errors='coerce')
    valid_rows[image_col] = pd.to_numeric(valid_rows[image_col], errors='coerce')
    valid_rows = valid_rows.dropna(subset=[livox_col, image_col])
    
    # 为每个livox点云（每行数据）单独处理，确保每个livox都能采样到2个图像
    # 首先收集所有可能的图像时间戳（在时间窗口内）
    livox_points = []  # 存储每个livox点云及其对应信息
    
    for idx, row in valid_rows.iterrows():
        try:
            livox_ts = float(row[livox_col])
            image_ts = float(row[image_col])
            seq_id = int(row[seq_col]) if seq_col in row and pd.notna(row[seq_col]) else None
            
            # 确定序列名称
            seq_name = None
            if seq_id is not None:
                seq_name = f"seq{seq_id:04d}"
                # 检查序列是否存在
                seq_dir = data_root / seq_name
                if not seq_dir.exists():
                    # 尝试其他格式
                    seq_name = f"seq{seq_id}"
            
            # 检查序列是否在指定的sequences中
            if seq_name and seq_name in sequences:
                livox_points.append({
                    'livox_ts': livox_ts,
                    'image_ts': image_ts,
                    'seq_name': seq_name,
                    'time_diff': abs(image_ts - livox_ts)
                })
        except (ValueError, TypeError, KeyError):
            continue
    
    # 收集所有序列的所有图像时间戳（从Image目录中读取所有图像文件）
    # 这比只从时间线文件读取更完整，可以找到时间窗口内的所有图像
    all_sequence_images = {}  # seq_name -> sorted list of image timestamps
    
    print("正在从Image目录读取所有图像时间戳...")
    for seq_name in sequences:
        seq_image_dir = data_root / seq_name / "Image"
        if seq_image_dir.exists():
            image_files = list(seq_image_dir.glob("*.png"))
            image_timestamps = []
            for img_file in image_files:
                try:
                    img_ts = float(img_file.stem)
                    image_timestamps.append(img_ts)
                except ValueError:
                    continue
            if image_timestamps:
                all_sequence_images[seq_name] = sorted(image_timestamps)
                print(f"  序列 {seq_name}: 找到 {len(image_timestamps)} 个图像文件")
    
    if not all_sequence_images:
        print("警告: 未找到任何图像文件，无法进行采样")
        return None
    
    # 为每个livox点云独立采样image_ratio个图像
    # 使用每个livox点云的索引作为唯一标识，确保每个livox都独立处理
    livox_groups = {}
    # 使用更大的搜索窗口来查找额外图像（默认0.5秒，确保能找到足够的图像）
    search_window_size = image_search_window
    
    for i, point in enumerate(livox_points):
        livox_ts = point['livox_ts']
        seq_name = point['seq_name']
        initial_image_ts = point['image_ts']
        
        # 为每个livox创建独立的组（使用livox索引+时间戳确保唯一性）
        group_key = (i, livox_ts)  # 使用索引和时间戳确保每个livox独立
        livox_groups[group_key] = []
        
        # 先添加时间线文件中对应的图像
        livox_groups[group_key].append((seq_name, initial_image_ts, abs(initial_image_ts - livox_ts)))
        
        # 如果还需要更多图像，在更大的搜索窗口内查找
        if len(livox_groups[group_key]) < image_ratio and seq_name in all_sequence_images:
            window_start = livox_ts - search_window_size
            window_end = livox_ts + search_window_size
            
            # 找到时间窗口内的所有图像（排除已添加的）
            existing_images = {initial_image_ts}
            window_images = []
            
            for img_ts in all_sequence_images[seq_name]:
                if window_start <= img_ts <= window_end and img_ts not in existing_images:
                    window_images.append((seq_name, img_ts, abs(img_ts - livox_ts)))
            
            # 按时间差排序，选择最接近的
            window_images.sort(key=lambda x: x[2])
            
            # 补充到image_ratio个
            needed = image_ratio - len(livox_groups[group_key])
            added_count = 0
            for img_info in window_images[:needed]:
                livox_groups[group_key].append(img_info)
                existing_images.add(img_info[1])  # 记录已添加的图像
                added_count += 1
            
            # 如果无法补充到足够的图像，至少保留已有的图像
            if added_count < needed and len(livox_groups[group_key]) < image_ratio:
                # 时间窗口内没有足够的图像，使用已有的图像（可能是1个）
                pass
    
    # 按livox时间戳排序
    sorted_livox_keys = sorted(livox_groups.keys(), key=lambda x: x[1])  # 按时间戳排序
    
    # 对于每个livox组，采样最多image_ratio个图像
    # 新策略：检测所有图像，不再使用关键帧选择
    livox_count = 0
    
    for group_key in sorted_livox_keys:
        images = livox_groups[group_key]
        if images:
            # 按时间差排序，选择最接近的image_ratio个图像
            images.sort(key=lambda x: x[2])  # 按时间差排序
            sampled = images[:image_ratio]  # 最多采样image_ratio个
            
            # 将所有采样的图像都加入采样集合（检测所有图像）
            for seq_name, img_ts, _ in sampled:
                sampled_image_timestamps.add((seq_name, img_ts))
            
            livox_count += 1
    
    # 统计采样情况
    livox_with_1_image = 0
    livox_with_2_images = 0
    for group_key in livox_groups:
        if len(livox_groups[group_key]) == 1:
            livox_with_1_image += 1
        elif len(livox_groups[group_key]) >= 2:
            livox_with_2_images += 1
    
    print(f"从 {livox_count} 个livox点云采样了 {len(sampled_image_timestamps)} 个图像时间戳（目标比例：1:{image_ratio}）")
    print(f"  其中 {livox_with_2_images} 个livox采样到2个图像，{livox_with_1_image} 个livox只采样到1个图像")
    if livox_with_1_image > 0:
        print(f"  警告: {livox_with_1_image} 个livox在搜索窗口（{search_window_size}秒）内无法找到足够的图像")
    
    # 新策略：检测所有图像，不再使用关键帧选择
    print(f"将检测所有 {len(sampled_image_timestamps)} 个图像（不再使用关键帧选择）")
    linked_count = 0
    # 创建时间戳到文件名的映射，用于后续关键帧筛选
    timestamp_to_filename = {}
    for seq_name, img_ts in sampled_image_timestamps:
        seq_image_dir = data_root / seq_name / "Image"
        if not seq_image_dir.exists():
            continue
        
        # 查找最接近时间戳的图像文件
        image_files = list(seq_image_dir.glob("*.png"))
        if not image_files:
            continue
        
        # 找到时间戳最接近的图像
        closest_file = None
        min_diff = float('inf')
        
        for img_file in image_files:
            try:
                file_ts = float(img_file.stem)
                diff = abs(file_ts - img_ts)
                if diff < min_diff:
                    min_diff = diff
                    closest_file = img_file
            except ValueError:
                continue
        
        if closest_file and min_diff < 0.1:  # 时间差小于0.1秒
            # 创建符号链接
            link_name = f"{seq_name}_{closest_file.name}"
            link_path = temp_image_dir / link_name
            
            if not link_path.exists():
                try:
                    link_path.symlink_to(closest_file.resolve())
                    linked_count += 1
                except OSError:
                    # 如果符号链接失败，复制文件
                    shutil.copy2(closest_file, link_path)
                    linked_count += 1
    
    print(f"已链接 {linked_count} 个采样图像到临时目录: {temp_image_dir}")
    
    if linked_count == 0:
        print("警告: 未找到任何采样图像")
        return None
    
    return temp_image_dir


def run_image_detection(args):
    """运行图像检测（YOLO11），支持基于时间线文件的图像采样。"""
    if args.skip_image_detection:
        print("跳过图像检测步骤")
        return
    
    print("=" * 60)
    print("步骤 2: 图像检测 (YOLO11)")
    print("=" * 60)
    
    # 删除上一次的检测结果目录，确保每次运行都使用一致的目录名
    previous_output_dir = Path(args.yolo_project) / args.yolo_name
    if previous_output_dir.exists():
        print(f"清理上一次的检测结果目录: {previous_output_dir}")
        try:
            shutil.rmtree(previous_output_dir)
            print(f"✅ 已删除: {previous_output_dir}")
        except Exception as e:
            print(f"警告: 删除目录失败: {e}")
            print(f"      请手动删除: {previous_output_dir}")
    
    detect_script = Path(__file__).parent.parent / "yolo11" / "detect_mmaud.py"
    
    if not detect_script.exists():
        print(f"警告: 未找到图像检测脚本: {detect_script}")
        print("请手动运行 detect_mmaud.py")
        return
    
    # 尝试基于时间线文件采样图像
    source_dir = None
    data_root_path = Path(args.data_root)
    
    # 确定时间线目录 - 使用与run_alignment相同的逻辑来确定data_split
    timeline_dir = None
    data_split = None  # 'train' 或 'val' 或 'test' 或 None
    
    # 确定当前处理的是训练集、验证集还是测试集（与run_alignment逻辑一致）
    if (data_root_path / "train").exists() or (data_root_path / "val").exists() or (data_root_path / "test").exists():
        # 情况1: 数据在 data_root/train/ 或 data_root/val/ 或 data_root/test/ 下
        if (data_root_path / "train").exists():
            data_split = "train"
        elif (data_root_path / "val").exists():
            data_split = "val"
        elif (data_root_path / "test").exists():
            data_split = "test"
    elif data_root_path.name in ["train", "val", "test"]:
        # 情况2: data_root 本身就是 train/val/test 目录（如 /home/p/MMUAV/test）
        data_split = data_root_path.name
    else:
        # 情况3: 其他情况，默认为 train
        data_split = "train"
    
    # 根据data_split确定时间线目录
    if args.output_dir:
        output_dir_path = Path(args.output_dir)
        # 优先使用对应的data_split目录（确保训练集和验证集使用正确的时间线）
        timeline_dir_candidate = output_dir_path / data_split
        if timeline_dir_candidate.exists():
            timeline_dir = timeline_dir_candidate
        else:
            # 如果对应目录不存在，使用output_dir本身（可能在后续步骤中创建）
            timeline_dir = output_dir_path
            print(f"警告: 时间线目录 {timeline_dir_candidate} 不存在，使用 {timeline_dir}")
    else:
        # 默认输出到 out/train/ 或 out/val/
        timeline_dir = Path("out") / data_split
    
    print(f"确定时间线目录: {timeline_dir} (data_split: {data_split}, data_root: {data_root_path})")
    
    # 新策略：以图像为主，按时间窗口采样图像
    print(f"按时间窗口采样图像（时间窗口: {args.window_size}秒，每窗口{args.frames_per_window}帧）...")
    sampled_dir = sample_images_by_time_window(
        data_root=data_root_path,
        sequences=args.sequences,
        window_size=args.window_size,
        frames_per_window=args.frames_per_window
    )
    if sampled_dir and sampled_dir.exists():
        source_dir = str(sampled_dir)
        print(f"✅ 已采样图像，使用采样后的图像目录: {source_dir}")
    
    # 如果没有成功采样，回退到原始逻辑
    if not source_dir:
    if args.yolo_source:
        source_dir = args.yolo_source
    else:
            # 自动查找序列目录下的Image目录
            # 查找指定序列目录下的Image
            image_dirs = []
            for seq_name in args.sequences:
                seq_image_dir = data_root_path / seq_name / "Image"
                if seq_image_dir.exists() and seq_image_dir.is_dir():
                    image_dirs.append(seq_image_dir)
                else:
                    print(f"警告: 序列 {seq_name} 的Image目录不存在: {seq_image_dir}")
            
            if not image_dirs:
                print(f"错误: 在 {data_root_path} 下未找到指定序列的Image目录: {args.sequences}")
                print("请使用 --yolo-source 参数指定图像目录，或检查 --sequences 参数")
                return
            
            # 如果找到多个序列的Image目录，创建一个临时汇总目录
            if len(image_dirs) > 1:
                # 创建临时汇总目录
                temp_image_dir = data_root_path / ".temp_all_images"
                if temp_image_dir.exists():
                    shutil.rmtree(temp_image_dir)
                temp_image_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"发现 {len(image_dirs)} 个序列的Image目录，正在汇总到临时目录...")
                file_count = 0
                seen_files = {}  # 用于跟踪已使用的文件名
                for img_dir in image_dirs:
                    seq_name = img_dir.parent.name
                    # 创建符号链接，处理文件名冲突
                    for img_file in img_dir.glob("*.png"):
                        base_name = img_file.name
                        # 检查文件名是否已被使用
                        if base_name in seen_files:
                            # 如果已存在，添加序列前缀
                            new_name = f"{seq_name}_{base_name}"
                            link_path = temp_image_dir / new_name
                        else:
                            link_path = temp_image_dir / base_name
                        
                        if not link_path.exists():
                            try:
                                link_path.symlink_to(img_file.resolve())
                                seen_files[base_name] = link_path
                                file_count += 1
                            except OSError:
                                # 如果符号链接失败（如Windows），则复制文件
                                shutil.copy2(img_file, link_path)
                                seen_files[base_name] = link_path
                                file_count += 1
                source_dir = str(temp_image_dir)
                print(f"图像已汇总到: {source_dir} (共 {file_count} 个文件)")
            else:
                source_dir = str(image_dirs[0])
                print(f"使用图像目录: {source_dir}")
    
    if not source_dir:
        print("错误: 无法确定图像源目录")
        return
    
    print(f"使用图像目录进行检测: {source_dir}")
    
    cmd = [
        sys.executable,
        str(detect_script),
        "--weights", args.yolo_weights,
        "--source", source_dir,
        "--conf-thres", str(args.yolo_conf_thres),
        "--iou-thres", "0.45",  # NMS阈值，可以适当提高以减少重复检测
        "--max-det", "30",  # 每张图像最大检测数
        "--project", args.yolo_project,
        "--name", args.yolo_name,
        "--batch", str(args.yolo_batch_size),  # 批处理加速
        "--num-kf", "0",  # 新策略：检测所有图像，不再使用关键帧选择
        "--target-class", "Drone",  # 只检测无人机类别
        "--save-conf",  # 保存置信度信息，便于调试
        "--roi-width", "1048",  # ROI区域宽度（从右往左0-1048）
        "--roi-height", "936",  # ROI区域高度（从下往上36-936）
        # 不添加 --verbose，使用默认的 verbose=False 以减少 I/O 阻塞
    ]
    
    # 新策略：不再使用关键帧选择，检测所有图像
    # 移除关键帧时间戳文件的传递
    
    print(f"⚠️  检测参数说明:")
    print(f"  置信度阈值: {args.yolo_conf_thres} (如果无人机检测不到，可以降低到0.25-0.3)")
    print(f"  目标类别: Drone (只检测无人机)")
    print(f"  ROI区域: 从右往左0-1048，从下往上36-936 (只检测该区域内的目标)")
    print(f"  如果检测到树干等误检，说明置信度阈值过低，可以适当提高")
    
    if args.yolo_half:
        cmd.append("--half")  # FP16半精度推理
    
    if args.yolo_save_crop:
        cmd.append("--save-crop")
    
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"⚠️  警告: 图像检测返回了非零退出码 ({result.returncode})，请检查上方输出")
        print("   提示: 这可能是正常的，如果检测过程已正常完成并生成了结果文件")
    else:
        print("图像检测完成")
        # 设置检测结果目录
        selected_crop_dir = Path(args.yolo_project) / args.yolo_name / "selected" / "crop"
        default_crops_dir = Path(args.yolo_project) / args.yolo_name / "crops"
        if selected_crop_dir.exists():
            args.detection_image_dir = str(selected_crop_dir)
            print(f"检测结果保存在: {selected_crop_dir}")
        elif default_crops_dir.exists():
            args.detection_image_dir = str(default_crops_dir)
            print(f"检测结果（裁剪目录）保存在: {default_crops_dir}")
        else:
            print("警告: 未找到 selected/crop 或 crops 目录，后续特征提取可能失败")


def run_pointcloud_detection(args):
    """运行点云检测（Livox Avia Detector）。"""
    if args.skip_pointcloud_detection:
        print("跳过点云检测步骤")
        return
    
    if not args.livox_detector_model:
        print("错误: 点云检测需要提供 --livox-detector-model 参数")
        return
    
    print("=" * 60)
    print("步骤 3: 点云检测 (Livox Avia Detector)")
    print("=" * 60)
    
    # 使用Python直接调用检测器
    import sys
    import os
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from point_cloud_processing.training_lidar.livox_avia_detector import LivoxAviaDetector
        
        data_root = Path(args.data_root)
        detector_model = Path(args.livox_detector_model)
        
        if not detector_model.exists():
            print(f"错误: 未找到检测器模型: {detector_model}")
            return
        
        # 只处理指定的序列
        all_metadata_rows = []
        detector = LivoxAviaDetector(str(detector_model))
        
        # 检查数据是在data_root下还是在data_root/train或data_root/val下
        if (data_root / "train").exists() or (data_root / "val").exists():
            # 对train和val分别处理指定序列
        for split in ['train', 'val']:
            split_dir = data_root / split
            if not split_dir.exists():
                continue
            
                print(f"处理 {split} 集中的指定序列: {args.sequences}")
                
                # 为每个指定序列单独处理
                split_results = {}
                for seq_name in args.sequences:
                    seq_dir = split_dir / seq_name
                    if not seq_dir.exists():
                        print(f"警告: 序列目录不存在: {seq_dir}")
                        continue
                    
                    print(f"  处理序列: {seq_name}")
                    # 直接处理单个序列目录
                    try:
                        samples = detector.detect_sequence_directory(
                            str(seq_dir),
                            livox_subdir="livox_avia"
                        )
                        split_results[seq_name] = samples
                        
                        # 保存检测结果
                        if args.livox_output_subdir:
                            output_dir = seq_dir / args.livox_output_subdir
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            for index, sample in enumerate(samples):
                                timestamp_start = sample.timestamps[0]
                                timestamp_end = sample.timestamps[-1]
                                filename = f"{timestamp_start}_{timestamp_end}_cluster{sample.cluster_id:03d}_{index:04d}.npy"
                                output_path = output_dir / filename
                                np.save(str(output_path), sample.points.astype(np.float32))
                                
                                # 添加元数据行
                                relative_path = f"{seq_name}/{args.livox_output_subdir}/{filename}"
                                metadata_row = f"{seq_name},{timestamp_start},{timestamp_end},{sample.cluster_id},{sample.score:.6f},{relative_path}\n"
                                all_metadata_rows.append(metadata_row)
                    except FileNotFoundError as e:
                        print(f"  警告: 序列 {seq_name} 处理失败: {e}")
                        continue
                
                print(f"{split} 集检测完成，共检测到 {sum(len(v) for v in split_results.values())} 个目标")
                    else:
            # 数据直接在data_root下，按序列处理
            print(f"处理指定序列: {args.sequences}")
            
            # 为每个指定序列单独处理
            all_results = {}
            for seq_name in args.sequences:
                seq_dir = data_root / seq_name
                if not seq_dir.exists():
                    print(f"警告: 序列目录不存在: {seq_dir}")
                    continue
                
                print(f"处理序列: {seq_name}")
                try:
                    samples = detector.detect_sequence_directory(
                        str(seq_dir),
                        livox_subdir="livox_avia"
                    )
                    all_results[seq_name] = samples
                    
                    # 保存检测结果
                    if args.livox_output_subdir:
                        output_dir = seq_dir / args.livox_output_subdir
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        for index, sample in enumerate(samples):
                            timestamp_start = sample.timestamps[0]
                            timestamp_end = sample.timestamps[-1]
                            filename = f"{timestamp_start}_{timestamp_end}_cluster{sample.cluster_id:03d}_{index:04d}.npy"
                            output_path = output_dir / filename
                            np.save(str(output_path), sample.points.astype(np.float32))
                            
                            # 添加元数据行
                            relative_path = f"{seq_name}/{args.livox_output_subdir}/{filename}"
                            metadata_row = f"{seq_name},{timestamp_start},{timestamp_end},{sample.cluster_id},{sample.score:.6f},{relative_path}\n"
                            all_metadata_rows.append(metadata_row)
                except FileNotFoundError as e:
                    print(f"警告: 序列 {seq_name} 处理失败: {e}")
                    continue
            
            print(f"所有序列检测完成，共检测到 {sum(len(v) for v in all_results.values())} 个目标")
        
        # 合并所有元数据到一个文件
        if all_metadata_rows:
            # 添加CSV头部
            csv_header = "sequence_name,timestamp_start,timestamp_end,cluster_id,score,points_path\n"
            all_metadata_rows.insert(0, csv_header)
            
            # 保存元数据文件
            # 如果数据在data_root/train或data_root/val下，保存到相应目录
            if (data_root / "train").exists() or (data_root / "val").exists():
                # 保存到第一个存在的split目录
                if (data_root / "train").exists():
                    metadata_path = data_root / "train" / args.livox_metadata_filename
                else:
                    metadata_path = data_root / "val" / args.livox_metadata_filename
            else:
                metadata_path = data_root / args.livox_metadata_filename
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.writelines(all_metadata_rows)
            args.detection_metadata_csv = str(metadata_path)
            print(f"检测元数据保存在: {metadata_path}")
        
        print("点云检测完成")
        
    except ImportError as e:
        print(f"错误: 无法导入点云检测器: {e}")
        print("请确保点云检测器模块可用")
    except Exception as e:
        print(f"错误: 点云检测失败: {e}")
        import traceback
        traceback.print_exc()


def run_lidar360_detection(args):
    """运行 Lidar 360 点云检测。"""
    if not args.lidar360_detector_model:
        print("跳过 Lidar 360 检测（未提供模型路径）")
        return
    
    print("=" * 60)
    print("步骤 3.1: Lidar 360 点云检测")
    print("=" * 60)
    
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from point_cloud_processing.training_lidar.lidar_360_detector import Lidar360Detector
        
        data_root = Path(args.data_root)
        detector_model = Path(args.lidar360_detector_model)
        
        if not detector_model.exists():
            print(f"错误: 未找到检测器模型: {detector_model}")
            return
        
        all_metadata_rows = []
        detector = Lidar360Detector(str(detector_model))
        
        # 处理所有指定序列
        if (data_root / "train").exists() or (data_root / "val").exists():
            for split in ['train', 'val']:
                split_dir = data_root / split
                if not split_dir.exists():
                    continue
                
                print(f"处理 {split} 集中的指定序列: {args.sequences}")
                
                for seq_name in args.sequences:
                    seq_dir = split_dir / seq_name
                    if not seq_dir.exists():
                        print(f"警告: 序列目录不存在: {seq_dir}")
                        continue
                    
                    print(f"  处理序列: {seq_name}")
                    try:
                        samples = detector.detect_sequence_directory(
                            str(seq_dir),
                            lidar_subdir="lidar_360"
                        )
                        
                        if args.lidar360_output_subdir:
                            output_dir = seq_dir / args.lidar360_output_subdir
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            for index, sample in enumerate(samples):
                                timestamp_start = sample.timestamps[0]
                                timestamp_end = sample.timestamps[-1]
                                filename = f"{timestamp_start}_{timestamp_end}_cluster{sample.cluster_id:03d}_{index:04d}.npy"
                                output_path = output_dir / filename
                                np.save(str(output_path), sample.points.astype(np.float32))
                                
                                relative_path = f"{seq_name}/{args.lidar360_output_subdir}/{filename}"
                                metadata_row = f"{seq_name},{timestamp_start},{timestamp_end},{sample.cluster_id},{sample.score:.6f},{relative_path}\n"
                                all_metadata_rows.append(metadata_row)
                    except FileNotFoundError as e:
                        print(f"  警告: 序列 {seq_name} 处理失败: {e}")
                        continue
                
                print(f"{split} 集检测完成")
        else:
            print(f"处理指定序列: {args.sequences}")
            
            for seq_name in args.sequences:
                seq_dir = data_root / seq_name
                if not seq_dir.exists():
                    print(f"警告: 序列目录不存在: {seq_dir}")
                    continue
                
                print(f"处理序列: {seq_name}")
                try:
                    samples = detector.detect_sequence_directory(
                        str(seq_dir),
                        lidar_subdir="lidar_360"
                    )
                    
                    if args.lidar360_output_subdir:
                        output_dir = seq_dir / args.lidar360_output_subdir
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        for index, sample in enumerate(samples):
                            timestamp_start = sample.timestamps[0]
                            timestamp_end = sample.timestamps[-1]
                            filename = f"{timestamp_start}_{timestamp_end}_cluster{sample.cluster_id:03d}_{index:04d}.npy"
                            output_path = output_dir / filename
                            np.save(str(output_path), sample.points.astype(np.float32))
                            
                            relative_path = f"{seq_name}/{args.lidar360_output_subdir}/{filename}"
                            metadata_row = f"{seq_name},{timestamp_start},{timestamp_end},{sample.cluster_id},{sample.score:.6f},{relative_path}\n"
                            all_metadata_rows.append(metadata_row)
                except FileNotFoundError as e:
                    print(f"警告: 序列 {seq_name} 处理失败: {e}")
                    continue
        
        # 保存元数据
        if all_metadata_rows:
            csv_header = "sequence_name,timestamp_start,timestamp_end,cluster_id,score,points_path\n"
            all_metadata_rows.insert(0, csv_header)
            
            if (data_root / "train").exists():
                metadata_path = data_root / "train" / args.lidar360_metadata_filename
            else:
                metadata_path = data_root / args.lidar360_metadata_filename
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.writelines(all_metadata_rows)
            print(f"检测元数据保存在: {metadata_path}")
        
        print("Lidar 360 检测完成")
        
    except ImportError as e:
        print(f"错误: 无法导入 Lidar 360 检测器: {e}")
    except Exception as e:
        print(f"错误: Lidar 360 检测失败: {e}")
        import traceback
        traceback.print_exc()


def run_radar_detection(args):
    """运行 Radar Enhance 点云检测。"""
    if not args.radar_detector_model:
        print("跳过 Radar Enhance 检测（未提供模型路径）")
        return
    
    print("=" * 60)
    print("步骤 3.2: Radar Enhance 点云检测")
    print("=" * 60)
    
    import sys
    # 使用当前目录下的radar_detector.py
    preprocessing_dir = Path(__file__).parent
    sys.path.insert(0, str(preprocessing_dir))
    
    try:
        from radar_detector import RadarCenterSequenceDetector
        
        data_root = Path(args.data_root)
        detector_model = Path(args.radar_detector_model)
        
        if not detector_model.exists():
            print(f"错误: 未找到检测器模型: {detector_model}")
            return
        
        # 自动查找scaler文件
        scaler_path = None
        # 尝试从模型路径推断scaler路径
        scaler_candidates = [
            str(detector_model).replace('.pth', '_scaler.pkl'),
            str(detector_model.parent / 'lstm_radar_enhance_model_scaler.pkl'),
            str(detector_model.parent.parent / 'lstm_radar_enhance_model_scaler.pkl'),
            'lstm_radar_enhance_model_scaler.pkl',  # 当前工作目录
        ]
        
        for candidate in scaler_candidates:
            if Path(candidate).exists():
                scaler_path = candidate
                print(f"✓ 找到scaler文件: {scaler_path}")
                break
        
        if scaler_path is None:
            print(f"⚠ 未找到scaler文件，将尝试自动推断路径")
            print(f"  尝试的路径: {scaler_candidates}")
        
        all_metadata_rows = []
        # 使用用户指定的阈值（如果scaler文件已加载，可以使用更高的阈值如0.3-0.5）
        # 如果scaler文件未找到，概率会很低，需要降低阈值（如0.001）
        detector = RadarCenterSequenceDetector(
            str(detector_model),
            prob_threshold=args.radar_prob_threshold,
            scaler_path=scaler_path  # 显式传递scaler路径
        )
        print(f"使用检测概率阈值: {args.radar_prob_threshold}")
        
        # 处理所有指定序列
        if (data_root / "train").exists() or (data_root / "val").exists():
            for split in ['train', 'val']:
                split_dir = data_root / split
                if not split_dir.exists():
                    continue
                
                print(f"处理 {split} 集中的指定序列: {args.sequences}")
                
                for seq_name in args.sequences:
                    seq_dir = split_dir / seq_name
                    if not seq_dir.exists():
                        print(f"警告: 序列目录不存在: {seq_dir}")
                        continue
                    
                    print(f"  处理序列: {seq_name}")
                    try:
                        samples = detector.detect_sequence_directory(
                            str(seq_dir),
                            radar_subdir="radar_enhance_pcl",
                        )
                        
                        print(f"    检测到 {len(samples)} 个结果")
                        
                        if args.radar_output_subdir:
                            output_dir = seq_dir / args.radar_output_subdir
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            for index, sample in enumerate(samples):
                                timestamp_start = sample.timestamps[0]
                                timestamp_end = sample.timestamps[-1]
                                filename = f"{timestamp_start}_{timestamp_end}_radar_cluster{sample.cluster_id}_{index:04d}.npy"
                                output_path = output_dir / filename
                                # Radar检测器输出的是点云聚类，直接保存点云数据
                                np.save(str(output_path), sample.points.astype(np.float32))
                                
                                relative_path = f"{seq_name}/{args.radar_output_subdir}/{filename}"
                                metadata_row = f"{seq_name},{timestamp_start},{timestamp_end},{sample.score:.6f},{relative_path},{sample.cluster_id}\n"
                                all_metadata_rows.append(metadata_row)
                    except FileNotFoundError as e:
                        print(f"  警告: 序列 {seq_name} 处理失败: {e}")
                        continue
                
                print(f"{split} 集检测完成")
        else:
            print(f"处理指定序列: {args.sequences}")
            
            for seq_name in args.sequences:
                seq_dir = data_root / seq_name
                if not seq_dir.exists():
                    print(f"警告: 序列目录不存在: {seq_dir}")
                    continue
                
                print(f"处理序列: {seq_name}")
                try:
                    samples = detector.detect_sequence_directory(
                        str(seq_dir),
                        radar_subdir="radar_enhance_pcl",
                    )
                    
                    print(f"  检测到 {len(samples)} 个结果")
                    
                    if args.radar_output_subdir:
                        output_dir = seq_dir / args.radar_output_subdir
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        for index, sample in enumerate(samples):
                            timestamp_start = sample.timestamps[0]
                            timestamp_end = sample.timestamps[-1]
                            filename = f"{timestamp_start}_{timestamp_end}_radar_cluster{sample.cluster_id}_{index:04d}.npy"
                            output_path = output_dir / filename
                            # Radar检测器输出的是点云聚类，直接保存点云数据
                            np.save(str(output_path), sample.points.astype(np.float32))
                            
                            relative_path = f"{seq_name}/{args.radar_output_subdir}/{filename}"
                            metadata_row = f"{seq_name},{timestamp_start},{timestamp_end},{sample.score:.6f},{relative_path},{sample.cluster_id}\n"
                            all_metadata_rows.append(metadata_row)
                except FileNotFoundError as e:
                    print(f"警告: 序列 {seq_name} 处理失败: {e}")
                    continue
        
        # 保存元数据
        if all_metadata_rows:
            csv_header = "sequence_name,timestamp_start,timestamp_end,score,points_path,cluster_id\n"
            all_metadata_rows.insert(0, csv_header)
            
            if (data_root / "train").exists():
                metadata_path = data_root / "train" / args.radar_metadata_filename
            else:
                metadata_path = data_root / args.radar_metadata_filename
            
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                f.writelines(all_metadata_rows)
            print(f"检测元数据保存在: {metadata_path} (共 {len(all_metadata_rows)-1} 条记录)")
        else:
            print("警告: 没有检测到任何Radar结果，未生成元数据文件")
            if (data_root / "train").exists():
                metadata_path = data_root / "train" / args.radar_metadata_filename
            else:
                metadata_path = data_root / args.radar_metadata_filename
            print(f"元数据文件应该保存在: {metadata_path}")
            print("提示: 可能是阈值过高或数据质量不足，可以尝试:")
            print(f"  1. 降低检测阈值（当前默认阈值: 0.6）")
            print(f"  2. 检查Radar数据是否存在且有效")
            print(f"  3. 检查检测器模型是否正确")
        
        print("Radar Enhance 检测完成")
        
    except ImportError as e:
        print(f"错误: 无法导入 Radar Enhance 检测器: {e}")
    except Exception as e:
        print(f"错误: Radar Enhance 检测失败: {e}")
        import traceback
        traceback.print_exc()


def _convert_radar_centers_to_pointcloud(centers: np.ndarray, num_points: int = 256) -> np.ndarray:
    """
    将Radar检测的中心序列转换为点云格式。
    
    Args:
        centers: 中心序列 (4, 3) - 4个中心点，每个点有xyz坐标
        num_points: 每个中心点周围生成的点数（默认256，总共1024点）
    
    Returns:
        pointcloud: (N, 3) 点云，其中 N = num_points * 4
    """
    if centers.shape[0] != 4:
        raise ValueError(f"期望4个中心点，但得到 {centers.shape[0]} 个")
    
    points_list = []
    # 在每个中心点周围生成点云（使用高斯噪声）
    for center in centers:
        # 在中心点周围生成点云
        # 使用小的高斯噪声（标准差0.1米）来模拟点云分布
        noise = np.random.normal(0, 0.1, (num_points, 3))
        center_points = center[None, :] + noise
        points_list.append(center_points)
    
    # 拼接所有点云
    pointcloud = np.concatenate(points_list, axis=0)
    return pointcloud.astype(np.float32)


def run_lidar360_pointnext_extraction(args):
    """运行 Lidar 360 点云特征提取（从检测结果）。"""
    if args.skip_pointnext or not args.lidar360_detector_model:
        print("跳过 Lidar 360 点云特征提取步骤")
        return True  # 跳过不算失败
    
    print("=" * 60)
    print("步骤 5.1: 提取 Lidar 360 点云特征 (PointNeXt)")
    print("=" * 60)
    
    extract_script = Path(__file__).parent.parent.parent / "point_cloud_processing" / "training_lidar" / "extract_pointnext_features.py"
    
    if not extract_script.exists():
        print(f"警告: 未找到点云特征提取脚本: {extract_script}")
        return
    
    # 构建元数据CSV路径
    data_root = Path(args.data_root)
    if (data_root / "train").exists():
        lidar360_metadata_path = data_root / "train" / args.lidar360_metadata_filename
    else:
        lidar360_metadata_path = data_root / args.lidar360_metadata_filename
    
    if not lidar360_metadata_path.exists():
        print(f"警告: 未找到 Lidar 360 检测元数据: {lidar360_metadata_path}")
        print("跳过 Lidar 360 特征提取")
        return True  # 缺少元数据时跳过，不算失败
    
    cmd = [
        sys.executable,
        str(extract_script),
        "--metadata-csv", str(lidar360_metadata_path),
        "--dataset-root", args.data_root,
        "--output", args.lidar360_pointnext_output,
    ]
    
    if args.pointnext_cfg:
        cmd.extend(["--cfg", args.pointnext_cfg])
    
    if args.pointnext_pretrained:
        cmd.extend(["--pretrained-path", args.pointnext_pretrained])
    
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"错误: Lidar 360 点云特征提取失败（返回码: {result.returncode}），请检查输出")
        return False
    
    # 验证输出文件是否存在
    output_path = Path(args.lidar360_pointnext_output)
    if output_path.exists():
        print(f"✓ Lidar 360 点云特征提取完成: {output_path} (大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return True
    else:
        print(f"警告: Lidar 360 特征文件未生成: {output_path}，但这不是必需的")
        return True  # Lidar 360是可选的


def run_radar_pointnext_extraction(args):
    """运行 Radar Enhance 点云特征提取（从检测结果，直接使用点云聚类）。"""
    if args.skip_pointnext or not args.radar_detector_model:
        print("跳过 Radar Enhance 点云特征提取步骤")
        return True  # 跳过不算失败
    
    print("=" * 60)
    print("步骤 5.2: 提取 Radar Enhance 点云特征 (PointNeXt)")
    print("=" * 60)
    
    extract_script = Path(__file__).parent.parent.parent / "point_cloud_processing" / "training_lidar" / "extract_pointnext_features.py"
    
    if not extract_script.exists():
        print(f"警告: 未找到点云特征提取脚本: {extract_script}")
        return
    
    # 构建元数据CSV路径
    data_root = Path(args.data_root)
    if (data_root / "train").exists():
        radar_metadata_path = data_root / "train" / args.radar_metadata_filename
    else:
        radar_metadata_path = data_root / args.radar_metadata_filename
    
    if not radar_metadata_path.exists():
        print(f"警告: 未找到 Radar Enhance 检测元数据: {radar_metadata_path}")
        print("跳过 Radar Enhance 特征提取")
        return True  # 缺少元数据时跳过，不算失败
    
    # 读取Radar检测元数据
    import pandas as pd
    metadata_df = pd.read_csv(radar_metadata_path)
    
    # 检查元数据列名（兼容旧格式和新格式）
    if 'points_path' in metadata_df.columns:
        points_path_col = 'points_path'
    elif 'centers_path' in metadata_df.columns:
        # 旧格式：需要转换
        print("检测到旧格式元数据（centers_path），需要转换为点云...")
        points_path_col = 'centers_path'
    else:
        print("错误: 元数据中未找到 points_path 或 centers_path 列")
        return
    
    # 创建临时点云目录（如果需要转换）
    temp_pc_dir = None
    temp_metadata_path = None
    
    if points_path_col == 'centers_path':
        # 旧格式：需要转换中心序列为点云
        temp_pc_dir = Path(args.output_dir) / "temp_radar_pointclouds"
        temp_pc_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"转换 {len(metadata_df)} 个中心序列为点云...")
        
        converted_paths = []
        for idx, row in metadata_df.iterrows():
            centers_path = Path(args.data_root) / row['centers_path']
            
            # 尝试在train和val中查找
            if not centers_path.exists():
                for split in ['train', 'val']:
                    alt_path = Path(args.data_root) / split / row['centers_path']
                    if alt_path.exists():
                        centers_path = alt_path
                        break
            
            if not centers_path.exists():
                print(f"警告: 未找到中心序列文件: {row['centers_path']}")
                continue
            
            # 加载中心序列
            centers = np.load(centers_path)
            
            # 转换为点云
            pointcloud = _convert_radar_centers_to_pointcloud(centers, num_points=256)
            
            # 保存为临时点云文件
            temp_pc_path = temp_pc_dir / f"radar_{idx:06d}.npy"
            np.save(temp_pc_path, pointcloud)
            converted_paths.append(str(temp_pc_path))
        
        print(f"已转换 {len(converted_paths)} 个中心序列为点云")
        
        # 创建临时元数据CSV（使用临时点云路径）
        temp_metadata_path = temp_pc_dir / "temp_radar_metadata.csv"
        with open(temp_metadata_path, 'w', encoding='utf-8') as f:
            f.write("sequence_name,timestamp_start,timestamp_end,score,points_path\n")
            for idx, (_, row) in enumerate(metadata_df.iterrows()):
                temp_pc_path = temp_pc_dir / f"radar_{idx:06d}.npy"
                f.write(f"{row['sequence_name']},{row['timestamp_start']},{row['timestamp_end']},{row['score']},{temp_pc_path}\n")
        
        metadata_csv_path = temp_metadata_path
    else:
        # 新格式：直接使用点云路径，但需要转换为绝对路径
        print(f"使用 {len(metadata_df)} 个点云聚类直接提取特征...")
        
        # 创建临时元数据CSV（转换为绝对路径）
        temp_pc_dir = Path(args.output_dir) / "temp_radar_pointclouds"
        temp_pc_dir.mkdir(parents=True, exist_ok=True)
        temp_metadata_path = temp_pc_dir / "temp_radar_metadata.csv"
        
        with open(temp_metadata_path, 'w', encoding='utf-8') as f:
            # 写入表头（兼容有无cluster_id列）
            if 'cluster_id' in metadata_df.columns:
                f.write("sequence_name,timestamp_start,timestamp_end,score,points_path,cluster_id\n")
            else:
                f.write("sequence_name,timestamp_start,timestamp_end,score,points_path\n")
            
            for _, row in metadata_df.iterrows():
                points_path = Path(args.data_root) / row['points_path']
                
                # 尝试在train和val中查找
                if not points_path.exists():
                    for split in ['train', 'val']:
                        alt_path = Path(args.data_root) / split / row['points_path']
                        if alt_path.exists():
                            points_path = alt_path
                            break
                
                if not points_path.exists():
                    print(f"警告: 未找到点云文件: {row['points_path']}")
                    continue
                
                # 写入绝对路径
                if 'cluster_id' in metadata_df.columns:
                    f.write(f"{row['sequence_name']},{row['timestamp_start']},{row['timestamp_end']},{row['score']},{points_path},{row['cluster_id']}\n")
                else:
                    f.write(f"{row['sequence_name']},{row['timestamp_start']},{row['timestamp_end']},{row['score']},{points_path}\n")
        
        metadata_csv_path = temp_metadata_path
    
    # 使用点云目录提取特征
    cmd = [
        sys.executable,
        str(extract_script),
        "--metadata-csv", str(metadata_csv_path),
        "--output", args.radar_pointnext_output,
    ]
    
    if args.pointnext_cfg:
        cmd.extend(["--cfg", args.pointnext_cfg])
    
    if args.pointnext_pretrained:
        cmd.extend(["--pretrained-path", args.pointnext_pretrained])
    
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # 清理临时文件
    if temp_pc_dir and temp_pc_dir.exists():
        import shutil
        shutil.rmtree(temp_pc_dir)
    
    if result.returncode != 0:
        print(f"错误: Radar Enhance 点云特征提取失败（返回码: {result.returncode}），请检查输出")
        return False
    
    # 验证输出文件是否存在
    output_path = Path(args.radar_pointnext_output)
    if output_path.exists():
        print(f"✓ Radar Enhance 点云特征提取完成: {output_path} (大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB)")
        return True
    else:
        print(f"警告: Radar 特征文件未生成: {output_path}，但这不是必需的")
        return True  # Radar是可选的


def run_pointnext_extraction(args):
    """运行点云特征提取（从检测结果）。"""
    if args.skip_pointnext:
        print("跳过点云特征提取步骤")
        return
    
    print("=" * 60)
    print("步骤 5: 提取点云特征 (PointNeXt)")
    print("=" * 60)
    
    extract_script = Path(__file__).parent.parent.parent / "point_cloud_processing" / "training_lidar" / "extract_pointnext_features.py"
    
    if not extract_script.exists():
        print(f"错误: 未找到点云特征提取脚本: {extract_script}")
        print("请手动运行 extract_pointnext_features.py")
        return False
    
    cmd = [
        sys.executable,
        str(extract_script),
        "--output", args.pointnext_output,
    ]
    
    # 优先使用检测元数据CSV
    if args.detection_metadata_csv:
        cmd.extend(["--metadata-csv", args.detection_metadata_csv])
        # 需要指定数据集根目录（points_path是相对于序列目录的）
        # 注意：points_path在CSV中是相对于序列目录的，所以需要data_root
        cmd.extend(["--dataset-root", args.data_root])
    elif args.detection_pointcloud_dir:
        cmd.extend(["--point-cloud-dir", args.detection_pointcloud_dir])
    elif args.point_cloud_dir:
        cmd.extend(["--point-cloud-dir", args.point_cloud_dir])
    else:
        print("错误: 必须提供检测后的点云目录或元数据CSV")
        return False
    
    if args.pointnext_cfg:
        cmd.extend(["--cfg", args.pointnext_cfg])
    
    if args.pointnext_pretrained:
        cmd.extend(["--pretrained-path", args.pointnext_pretrained])
    
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"错误: 点云特征提取失败（返回码: {result.returncode}），请检查输出")
        print(f"  命令: {' '.join(cmd)}")
        return False
    
    # 验证输出文件是否存在
    output_path = Path(args.pointnext_output)
    if not output_path.exists():
        print(f"错误: 点云特征文件未生成: {output_path}")
        print(f"  请检查特征提取脚本是否成功执行")
        print(f"  输出目录是否存在: {output_path.parent}")
        return False
    
    print(f"✓ 点云特征提取完成: {output_path} (大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return True


def run_convnext_extraction(args):
    """运行图像特征提取（从检测结果）。"""
    if args.skip_convnext:
        print("跳过图像特征提取步骤")
        return True  # 跳过不算失败
    
    print("=" * 60)
    print("步骤 4: 提取图像特征 (ConvNeXt)")
    print("=" * 60)
    
    extract_script = Path(__file__).parent.parent / "yolo11" / "extract_convnext_features.py"
    
    if not extract_script.exists():
        print(f"错误: 未找到图像特征提取脚本: {extract_script}")
        print("请手动运行 extract_convnext_features.py")
        return False
    
    cmd = [
        sys.executable,
        str(extract_script),
        "--output", args.convnext_output,
    ]
    
    # 智能查找检测后的裁剪图像目录
    detection_image_dir = None
    
    # 1. 优先使用已设置的detection_image_dir
    if args.detection_image_dir and Path(args.detection_image_dir).exists():
        detection_image_dir = args.detection_image_dir
        print(f"使用已设置的检测图像目录: {detection_image_dir}")
    else:
        # 2. 尝试查找YOLO检测输出目录
        yolo_output_base = Path(args.yolo_project) / args.yolo_name
        possible_dirs = [
            yolo_output_base / "selected" / "crop",  # 关键帧裁剪目录（优先）
            yolo_output_base / "selected",  # 关键帧标注图像目录
            yolo_output_base / "crops",  # 所有裁剪图像目录
            yolo_output_base,  # 检测结果根目录
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                # 检查目录中是否有图像文件
                image_files = list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpg")) + \
                             list(dir_path.glob("*.jpeg"))
                if len(image_files) > 0:
                    detection_image_dir = str(dir_path)
                    args.detection_image_dir = detection_image_dir  # 保存设置
                    print(f"自动找到检测图像目录: {detection_image_dir} (包含 {len(image_files)} 张图像)")
                    break
    
    # 3. 如果还是找不到，使用用户提供的image_dir
    if not detection_image_dir and args.image_dir:
        if Path(args.image_dir).exists():
            detection_image_dir = args.image_dir
            print(f"使用用户提供的图像目录: {detection_image_dir}")
        else:
            print(f"警告: 用户提供的图像目录不存在: {args.image_dir}")
    
    # 4. 如果仍然找不到，使用数据根目录下的图像目录作为备用
    if not detection_image_dir:
        data_root = Path(args.data_root)
        fallback_dirs = [
            data_root / ".temp_all_images",  # 对齐后的图像目录
            data_root / "Image",  # 原始图像目录
        ]
        for dir_path in fallback_dirs:
            if dir_path.exists():
                image_files = list(dir_path.glob("*.png")) + list(dir_path.glob("*.jpg"))
                if len(image_files) > 0:
                    detection_image_dir = str(dir_path)
                    print(f"使用备用图像目录: {detection_image_dir} (包含 {len(image_files)} 张图像)")
                    print(f"  注意: 这是原始图像目录，不是检测后的裁剪图像")
                    break
    
    # 5. 如果仍然找不到，报错
    if not detection_image_dir:
        print("错误: 无法找到检测后的图像目录")
        print(f"  已检查的位置:")
        print(f"    - {yolo_output_base / 'selected' / 'crop'}")
        print(f"    - {yolo_output_base / 'crops'}")
        print(f"    - {args.image_dir if args.image_dir else '(未提供)'}")
        print(f"  建议:")
        print(f"    1. 检查YOLO检测是否成功完成")
        print(f"    2. 确认使用了 --yolo-save-crop 参数")
        print(f"    3. 或使用 --image-dir 参数手动指定图像目录")
        return False
    
    cmd.extend(["--image-dir", detection_image_dir])
    
    print(f"运行: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"错误: 图像特征提取失败（返回码: {result.returncode}），请检查输出")
        print(f"  命令: {' '.join(cmd)}")
        return False
    
    # 验证输出文件是否存在
    output_path = Path(args.convnext_output)
    if not output_path.exists():
        print(f"错误: 图像特征文件未生成: {output_path}")
        print(f"  请检查特征提取脚本是否成功执行")
        print(f"  输出目录是否存在: {output_path.parent}")
        return False
    
    print(f"✓ 图像特征提取完成: {output_path} (大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return True


def align_features_by_image(
    pointnext_features_path: Path,
    convnext_features_path: Path,
    pointcloud_metadata_csv: Path,
    output_pointnext_path: Path,
    output_convnext_path: Path,
    window_size: float = 0.4,
    max_time_diff: float = 0.5
):
    """
    以图像检测结果为基准，在时间窗口内搜索点云特征。
    
    新策略：
    1. 以图像检测结果为基准（每个时间窗口内的图像）
    2. 在每个图像时间窗口内搜索是否有对应的点云检测结果
    3. 如果窗口内有点云检测，拼接点云特征；否则使用零向量
    
    参数:
        pointnext_features_path: 原始点云特征文件路径
        convnext_features_path: 原始图像特征文件路径
        pointcloud_metadata_csv: 点云检测元数据CSV文件路径
        output_pointnext_path: 对齐后的点云特征输出路径
        output_convnext_path: 对齐后的图像特征输出路径
        window_size: 时间窗口大小（秒，默认0.4秒）
        max_time_diff: 最大允许时间差（秒，用于搜索点云）
    """
    print("=" * 60)
    print("步骤 6: 基于图像检测的特征对齐（以图像为主）")
    print("=" * 60)
    
    # 读取图像特征（作为基准）
    convnext_data = torch.load(convnext_features_path, map_location='cpu')
    convnext_features = torch.tensor(convnext_data['features'])
    convnext_metadata = convnext_data.get('metadata', [])
    print(f"读取图像特征: {len(convnext_features)} 个特征（作为对齐基准）")
    
    # 读取点云特征
    pointnext_data = torch.load(pointnext_features_path, map_location='cpu')
    pointnext_features = pointnext_data['features']
    pointnext_metadata = pointnext_data.get('metadata', [])
    print(f"读取点云特征: {len(pointnext_features)} 个特征")
    
    # 读取点云检测元数据（用于匹配）
    pointcloud_timestamp_map = {}  # (seq_name, timestamp) -> (feature_idx, timestamp_start, timestamp_end)
    if pointcloud_metadata_csv.exists():
        metadata_df = pd.read_csv(pointcloud_metadata_csv)
        print(f"读取点云检测元数据: {len(metadata_df)} 个检测结果")
        
        # 构建点云检测的时间戳索引
        for idx, row in metadata_df.iterrows():
            seq_name = row.get('sequence_name', '')
            timestamp_start = float(row.get('timestamp_start', 0))
            timestamp_end = float(row.get('timestamp_end', timestamp_start))
            # 使用中点时间戳作为匹配基准
            pointcloud_timestamp = (timestamp_start + timestamp_end) / 2.0
            
            # 点云特征索引应该与CSV行索引一致
            if idx < len(pointnext_features):
                pointcloud_timestamp_map[(seq_name, pointcloud_timestamp)] = (idx, timestamp_start, timestamp_end)
    else:
        print(f"警告: 点云检测元数据文件不存在: {pointcloud_metadata_csv}")
        print("将使用点云特征元数据中的时间戳信息")
        # 尝试从点云特征元数据中提取时间戳
        for idx, meta in enumerate(pointnext_metadata):
            if isinstance(meta, dict):
                timestamp = meta.get('timestamp') or meta.get('timestamp_start')
                seq_name = meta.get('seq_name') or meta.get('sequence_name', '')
                if timestamp is not None:
                    pointcloud_timestamp_map[(seq_name, float(timestamp))] = (idx, float(timestamp), float(timestamp))
    
    print(f"构建点云时间戳索引: {len(pointcloud_timestamp_map)} 个有效时间戳")
    
    # 按时间窗口组织图像特征
    # 将图像特征按时间戳排序，然后按时间窗口分组
    image_windows = []  # 每个窗口包含 (window_start, window_end, image_indices, image_timestamps)
    image_timestamps_with_idx = []  # (timestamp, idx, metadata)
    
    for idx, meta in enumerate(convnext_metadata):
        timestamp = meta.get('timestamp')
        if timestamp is not None:
            image_timestamps_with_idx.append((float(timestamp), idx, meta))
    
    # 按时间戳排序
    image_timestamps_with_idx.sort(key=lambda x: x[0])
    
    # 按时间窗口分组图像
    if image_timestamps_with_idx:
        window_start = image_timestamps_with_idx[0][0]
        current_window_images = []
        
        for img_ts, img_idx, img_meta in image_timestamps_with_idx:
            if img_ts < window_start + window_size:
                current_window_images.append((img_ts, img_idx, img_meta))
            else:
                # 窗口已满，保存当前窗口
                if current_window_images:
                    image_windows.append((window_start, window_start + window_size, current_window_images))
                # 开始新窗口
                window_start = img_ts
                current_window_images = [(img_ts, img_idx, img_meta)]
        
        # 处理最后一个窗口
        if current_window_images:
            image_windows.append((window_start, window_start + window_size, current_window_images))
    
    print(f"按时间窗口组织图像特征: {len(image_windows)} 个时间窗口（窗口大小: {window_size}秒）")
    
    # 以图像窗口为基准，搜索点云特征
    aligned_pointnext_features = []
    aligned_convnext_features = []
    aligned_metadata = []
    
    matched_count = 0
    unmatched_count = 0
    
    for window_start, window_end, window_images in image_windows:
        # 计算窗口中心时间戳（用于匹配点云）
        window_center = (window_start + window_end) / 2.0
        
        # 在窗口内搜索点云检测结果
        best_pointcloud_idx = None
        best_time_diff = float('inf')
        best_pointcloud_timestamp = None
        
        for (seq_name, pc_timestamp), (pc_idx, pc_start, pc_end) in pointcloud_timestamp_map.items():
            # 检查点云时间戳是否在窗口内
            if window_start <= pc_timestamp <= window_end:
                time_diff = abs(pc_timestamp - window_center)
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_pointcloud_idx = pc_idx
                    best_pointcloud_timestamp = pc_timestamp
        
        # 聚合窗口内的图像特征（使用平均池化）
        if window_images:
            window_image_features = [convnext_features[img_idx] for _, img_idx, _ in window_images]
            aggregated_image_feature = torch.stack(window_image_features).mean(dim=0)  # 平均池化
            
            # 如果找到点云特征，使用点云特征；否则使用零向量
            if best_pointcloud_idx is not None:
                aligned_pointnext_features.append(pointnext_features[best_pointcloud_idx])
                matched_count += 1
            else:
                # 使用零向量作为点云特征
                if len(pointnext_features) > 0:
                    aligned_pointnext_features.append(torch.zeros_like(pointnext_features[0]))
                else:
                    # 如果点云特征为空，创建一个默认维度的零向量（512维，PointNeXt的输出维度）
                    aligned_pointnext_features.append(torch.zeros(512))
                unmatched_count += 1
            
            aligned_convnext_features.append(aggregated_image_feature)
            
            # 记录元数据
            window_image_timestamps = [img_ts for img_ts, _, _ in window_images]
            aligned_metadata.append({
                'window_start': window_start,
                'window_end': window_end,
                'window_center': window_center,
                'image_timestamps': window_image_timestamps,
                'image_count': len(window_images),
                'pointcloud_timestamp': best_pointcloud_timestamp,
                'time_diff': best_time_diff if best_pointcloud_idx is not None else None,
                'pointcloud_idx': best_pointcloud_idx,
                'has_pointcloud': best_pointcloud_idx is not None
            })
    
    print(f"特征对齐完成:")
    print(f"  图像窗口数: {len(image_windows)} 个")
    print(f"  匹配到点云: {matched_count} 个窗口")
    print(f"  未匹配点云: {unmatched_count} 个窗口（使用零向量）")
    
    # 保存对齐后的特征
    aligned_pointnext_tensor = torch.stack(aligned_pointnext_features)
    aligned_convnext_tensor = torch.stack(aligned_convnext_features)
    
    torch.save(
        {
            "features": aligned_pointnext_tensor,
            "metadata": aligned_metadata,
            "model": pointnext_data.get('model', 'pointnext'),
        },
        output_pointnext_path,
    )
    
    torch.save(
        {
            "features": aligned_convnext_tensor,
            "metadata": aligned_metadata,
            "model": "convnext_tiny",
            "normalize": convnext_data.get('normalize', False),
            "image_size": convnext_data.get('image_size', 224),
        },
        output_convnext_path,
    )
    
    print(f"对齐后的特征已保存:")
    print(f"  点云特征: {output_pointnext_path}")
    print(f"  图像特征: {output_convnext_path}")
    
    return True


def align_features_by_image_4modal(
    image_features_path: Path,
    livox_features_path: Path,
    lidar360_features_path: Optional[Path],
    radar_features_path: Optional[Path],
    image_metadata_csv: Optional[Path],
    livox_metadata_csv: Optional[Path],
    lidar360_metadata_csv: Optional[Path],
    radar_metadata_csv: Optional[Path],
    output_image_path: Path,
    output_livox_path: Path,
    output_lidar360_path: Optional[Path],
    output_radar_path: Optional[Path],
    window_size: float = 0.4,
    max_time_diff: float = 0.5,
):
    """
    以图像检测结果为基准（四模态版本），在时间窗口内搜索点云特征。
    
    策略：
    1. 以图像检测结果为基准（每个时间窗口内的图像）
    2. 在每个图像时间窗口内搜索是否有对应的点云检测结果（Livox、Lidar 360、Radar）
    3. 如果窗口内缺少某个点云模态，使用零向量
    
    参数:
        与align_features_multimodal相同
    """
    print("=" * 60)
    print("步骤 6: 基于图像检测的特征对齐（以图像为主，四模态）")
    print("=" * 60)
    
    # 读取图像特征（作为基准）
    image_data = torch.load(image_features_path, map_location='cpu')
    image_features = torch.tensor(image_data['features'])
    image_metadata_list = image_data.get('metadata', [])
    print(f"读取图像特征: {len(image_features)} 个特征（作为对齐基准）")
    
    # 读取点云特征
    livox_data = torch.load(livox_features_path, map_location='cpu')
    livox_features = livox_data['features']
    livox_metadata_list = livox_data.get('metadata', [])
    print(f"读取Livox点云特征: {len(livox_features)} 个特征")
    
    lidar360_features = None
    lidar360_metadata_list = []
    if lidar360_features_path and lidar360_features_path.exists():
        lidar360_data = torch.load(lidar360_features_path, map_location='cpu')
        lidar360_features = lidar360_data['features']
        lidar360_metadata_list = lidar360_data.get('metadata', [])
        print(f"读取Lidar 360点云特征: {len(lidar360_features)} 个特征")
    
    radar_features = None
    radar_metadata_list = []
    if radar_features_path and radar_features_path.exists():
        radar_data = torch.load(radar_features_path, map_location='cpu')
        radar_features = radar_data['features']
        radar_metadata_list = radar_data.get('metadata', [])
        print(f"读取Radar点云特征: {len(radar_features)} 个特征")
    
    # 构建点云时间戳索引
    livox_timestamps = []
    for idx, meta in enumerate(livox_metadata_list):
        timestamp = meta.get('timestamp') or meta.get('timestamp_start')
        if timestamp is not None:
            livox_timestamps.append(('livox', float(timestamp), idx, meta))
    
    lidar360_timestamps = []
    if lidar360_features is not None:
        for idx, meta in enumerate(lidar360_metadata_list):
            timestamp = meta.get('timestamp') or meta.get('timestamp_start')
            if timestamp is not None:
                lidar360_timestamps.append(('lidar360', float(timestamp), idx, meta))
    
    radar_timestamps = []
    if radar_features is not None:
        for idx, meta in enumerate(radar_metadata_list):
            timestamp = meta.get('timestamp') or meta.get('timestamp_start')
            if timestamp is not None:
                radar_timestamps.append(('radar', float(timestamp), idx, meta))
    
    # 构建图像时间戳索引
    image_timestamps = []
    for idx, meta in enumerate(image_metadata_list):
        timestamp = meta.get('timestamp')
        if timestamp is not None:
            image_timestamps.append((float(timestamp), idx, meta))
    
    image_timestamps.sort(key=lambda x: x[0])
    
    print(f"构建点云时间戳索引: Livox={len(livox_timestamps)}, Lidar 360={len(lidar360_timestamps)}, Radar={len(radar_timestamps)}")
    print(f"构建图像时间戳索引: {len(image_timestamps)} 个有效时间戳")
    
    # 按时间窗口组织图像特征
    image_windows = []
    if image_timestamps:
        window_start = image_timestamps[0][0]
        current_window_images = []
        
        for img_ts, img_idx, img_meta in image_timestamps:
            if img_ts < window_start + window_size:
                current_window_images.append((img_ts, img_idx, img_meta))
            else:
                if current_window_images:
                    image_windows.append((window_start, window_start + window_size, current_window_images))
                window_start = img_ts
                current_window_images = [(img_ts, img_idx, img_meta)]
        
        if current_window_images:
            image_windows.append((window_start, window_start + window_size, current_window_images))
    
    print(f"按时间窗口组织图像特征: {len(image_windows)} 个时间窗口（窗口大小: {window_size}秒）")
    
    # 以图像窗口为基准，搜索点云特征
    aligned_image_features = []
    aligned_livox_features = []
    aligned_lidar360_features = []
    aligned_radar_features = []
    aligned_metadata = []
    
    for window_start, window_end, window_images in image_windows:
        window_center = (window_start + window_end) / 2.0
        
        # 聚合窗口内的图像特征（平均池化）
        if window_images:
            window_image_features = [image_features[img_idx] for _, img_idx, _ in window_images]
            aggregated_image_feature = torch.stack(window_image_features).mean(dim=0)
        else:
            aggregated_image_feature = torch.zeros(image_features.shape[1]) if len(image_features) > 0 else torch.zeros(768)
        
        # 在窗口内搜索Livox特征
        best_livox_idx = None
        best_livox_time_diff = float('inf')
        for modality, pc_ts, pc_idx, pc_meta in livox_timestamps:
            if window_start <= pc_ts <= window_end:
                time_diff = abs(pc_ts - window_center)
                if time_diff < best_livox_time_diff:
                    best_livox_time_diff = time_diff
                    best_livox_idx = pc_idx
        
        if best_livox_idx is not None:
            aligned_livox_features.append(livox_features[best_livox_idx])
            livox_mask = 1.0
        else:
            aligned_livox_features.append(torch.zeros(livox_features.shape[1]) if len(livox_features) > 0 else torch.zeros(512))
            livox_mask = 0.0
        
        # 在窗口内搜索Lidar 360特征
        best_lidar360_idx = None
        best_lidar360_time_diff = float('inf')
        if lidar360_features is not None:
            for modality, pc_ts, pc_idx, pc_meta in lidar360_timestamps:
                if window_start <= pc_ts <= window_end:
                    time_diff = abs(pc_ts - window_center)
                    if time_diff < best_lidar360_time_diff:
                        best_lidar360_time_diff = time_diff
                        best_lidar360_idx = pc_idx
            
            if best_lidar360_idx is not None:
                aligned_lidar360_features.append(lidar360_features[best_lidar360_idx])
                lidar360_mask = 1.0
            else:
                aligned_lidar360_features.append(torch.zeros(lidar360_features.shape[1]))
                lidar360_mask = 0.0
        
        # 在窗口内搜索Radar特征
        best_radar_idx = None
        best_radar_time_diff = float('inf')
        if radar_features is not None:
            for modality, pc_ts, pc_idx, pc_meta in radar_timestamps:
                if window_start <= pc_ts <= window_end:
                    time_diff = abs(pc_ts - window_center)
                    if time_diff < best_radar_time_diff:
                        best_radar_time_diff = time_diff
                        best_radar_idx = pc_idx
            
            if best_radar_idx is not None:
                aligned_radar_features.append(radar_features[best_radar_idx])
                radar_mask = 1.0
            else:
                aligned_radar_features.append(torch.zeros(radar_features.shape[1]))
                radar_mask = 0.0
        
        aligned_image_features.append(aggregated_image_feature)
        
        # 记录元数据
        window_image_timestamps = [img_ts for img_ts, _, _ in window_images]
        aligned_metadata.append({
            'window_start': window_start,
            'window_end': window_end,
            'window_center': window_center,
            'image_timestamps': window_image_timestamps,
            'image_count': len(window_images),
            'image_mask': 1.0 if window_images else 0.0,
            'livox_mask': livox_mask,
            'lidar360_mask': lidar360_mask if lidar360_features is not None else 0.0,
            'radar_mask': radar_mask if radar_features is not None else 0.0,
        })
    
    print(f"特征对齐完成:")
    print(f"  图像窗口数: {len(image_windows)} 个")
    print(f"  Livox匹配: {sum(1 for m in aligned_metadata if m['livox_mask'] > 0)} 个窗口")
    if lidar360_features is not None:
        print(f"  Lidar 360匹配: {sum(1 for m in aligned_metadata if m['lidar360_mask'] > 0)} 个窗口")
    if radar_features is not None:
        print(f"  Radar匹配: {sum(1 for m in aligned_metadata if m.get('radar_mask', 0) > 0)} 个窗口")
    
    # 保存对齐后的特征
    aligned_image_tensor = torch.stack(aligned_image_features)
    aligned_livox_tensor = torch.stack(aligned_livox_features)
    
    torch.save(
        {
            "features": aligned_image_tensor,
            "metadata": aligned_metadata,
            "model": "convnext_tiny",
        },
        output_image_path,
    )
    
    torch.save(
        {
            "features": aligned_livox_tensor,
            "metadata": aligned_metadata,
            "model": "pointnext",
        },
        output_livox_path,
    )
    
    if lidar360_features is not None and output_lidar360_path:
        aligned_lidar360_tensor = torch.stack(aligned_lidar360_features)
        torch.save(
            {
                "features": aligned_lidar360_tensor,
                "metadata": aligned_metadata,
                "model": "pointnext",
            },
            output_lidar360_path,
        )
        print(f"  Lidar 360特征: {output_lidar360_path}")
    
    if radar_features is not None and output_radar_path:
        aligned_radar_tensor = torch.stack(aligned_radar_features)
        torch.save(
            {
                "features": aligned_radar_tensor,
                "metadata": aligned_metadata,
                "model": "pointnext",
            },
            output_radar_path,
        )
        print(f"  Radar特征: {output_radar_path}")
    
    print(f"对齐后的特征已保存:")
    print(f"  图像特征: {output_image_path}")
    print(f"  Livox特征: {output_livox_path}")
    
    return True


def align_features_by_pointcloud(
    image_features_path: Path,
    livox_features_path: Path,
    lidar360_features_path: Optional[Path],
    radar_features_path: Optional[Path],
    image_metadata_csv: Optional[Path],
    livox_metadata_csv: Optional[Path],
    lidar360_metadata_csv: Optional[Path],
    radar_metadata_csv: Optional[Path],
    output_image_path: Path,
    output_livox_path: Path,
    output_lidar360_path: Optional[Path],
    output_radar_path: Optional[Path],
    window_size: float = 0.4,
    max_time_diff: float = 0.5,
):
    """
    以点云检测结果为基准（Livox、Lidar 360或Radar），在时间窗口内搜索其他模态特征。
    
    策略：
    1. 以点云检测结果为基准（优先使用Livox，如果没有则使用Lidar 360或Radar）
    2. 在每个点云时间窗口内搜索是否有对应的图像和其他点云检测结果
    3. 如果窗口内缺少某个模态，使用零向量
    
    参数:
        image_features_path: 图像特征文件路径
        livox_features_path: Livox点云特征文件路径
        lidar360_features_path: Lidar 360点云特征文件路径（可选）
        radar_features_path: Radar点云特征文件路径（可选）
        image_metadata_csv: 图像检测元数据CSV文件路径（可选）
        livox_metadata_csv: Livox检测元数据CSV文件路径（可选）
        lidar360_metadata_csv: Lidar 360检测元数据CSV文件路径（可选）
        radar_metadata_csv: Radar检测元数据CSV文件路径（可选）
        output_image_path: 对齐后的图像特征输出路径
        output_livox_path: 对齐后的Livox特征输出路径
        output_lidar360_path: 对齐后的Lidar 360特征输出路径（可选）
        output_radar_path: 对齐后的Radar特征输出路径（可选）
        window_size: 时间窗口大小（秒，默认0.4秒）
        max_time_diff: 最大允许时间差（秒，用于搜索其他模态）
    """
    print("=" * 60)
    print("步骤 6: 基于点云检测的特征对齐（以点云为主）")
    print("=" * 60)
    
    # 读取点云特征（作为基准，优先使用Livox）
    livox_data = torch.load(livox_features_path, map_location='cpu')
    livox_features = torch.tensor(livox_data['features'])
    livox_metadata = livox_data.get('metadata', [])
    print(f"读取Livox点云特征: {len(livox_features)} 个特征（作为对齐基准）")
    
    lidar360_features = None
    lidar360_metadata = []
    if lidar360_features_path and lidar360_features_path.exists():
        lidar360_data = torch.load(lidar360_features_path, map_location='cpu')
        lidar360_features = lidar360_data['features']
        lidar360_metadata = lidar360_data.get('metadata', [])
        print(f"读取Lidar 360点云特征: {len(lidar360_features)} 个特征")
    
    radar_features = None
    radar_metadata = []
    if radar_features_path and radar_features_path.exists():
        radar_data = torch.load(radar_features_path, map_location='cpu')
        radar_features = radar_data['features']
        radar_metadata = radar_data.get('metadata', [])
        print(f"读取Radar点云特征: {len(radar_features)} 个特征")
    
    # 读取图像特征
    image_data = torch.load(image_features_path, map_location='cpu')
    image_features = torch.tensor(image_data['features'])
    image_metadata = image_data.get('metadata', [])
    print(f"读取图像特征: {len(image_features)} 个特征")
    
    # 构建点云时间戳索引（使用Livox作为主要基准）
    pointcloud_timestamps = []  # (timestamp, modality, feature_idx, metadata)
    for idx, meta in enumerate(livox_metadata):
        timestamp = meta.get('timestamp') or meta.get('timestamp_start')
        if timestamp is not None:
            pointcloud_timestamps.append(('livox', float(timestamp), idx, meta))
    
    # 如果Lidar 360存在，也添加到时间戳列表
    if lidar360_features is not None:
        for idx, meta in enumerate(lidar360_metadata):
            timestamp = meta.get('timestamp') or meta.get('timestamp_start')
            if timestamp is not None:
                pointcloud_timestamps.append(('lidar360', float(timestamp), idx, meta))
    
    # 如果Radar存在，也添加到时间戳列表
    if radar_features is not None:
        for idx, meta in enumerate(radar_metadata):
            timestamp = meta.get('timestamp') or meta.get('timestamp_start')
            if timestamp is not None:
                pointcloud_timestamps.append(('radar', float(timestamp), idx, meta))
    
    # 按时间戳排序
    pointcloud_timestamps.sort(key=lambda x: x[1])
    
    # 构建图像时间戳索引
    image_timestamps = []  # (timestamp, feature_idx, metadata)
    for idx, meta in enumerate(image_metadata):
        timestamp = meta.get('timestamp')
        if timestamp is not None:
            image_timestamps.append((float(timestamp), idx, meta))
    
    image_timestamps.sort(key=lambda x: x[0])
    
    print(f"构建点云时间戳索引: {len(pointcloud_timestamps)} 个有效时间戳（Livox、Lidar 360、Radar）")
    print(f"构建图像时间戳索引: {len(image_timestamps)} 个有效时间戳")
    
    # 按时间窗口组织点云特征（以点云为主）
    pointcloud_windows = []  # 每个窗口包含 (window_start, window_end, pointcloud_data)
    if pointcloud_timestamps:
        window_start = pointcloud_timestamps[0][1]
        current_window_pointclouds = []
        
        for modality, pc_ts, pc_idx, pc_meta in pointcloud_timestamps:
            if pc_ts < window_start + window_size:
                current_window_pointclouds.append((modality, pc_ts, pc_idx, pc_meta))
            else:
                # 窗口已满，保存当前窗口
                if current_window_pointclouds:
                    pointcloud_windows.append((window_start, window_start + window_size, current_window_pointclouds))
                # 开始新窗口
                window_start = pc_ts
                current_window_pointclouds = [(modality, pc_ts, pc_idx, pc_meta)]
        
        # 处理最后一个窗口
        if current_window_pointclouds:
            pointcloud_windows.append((window_start, window_start + window_size, current_window_pointclouds))
    
    print(f"按时间窗口组织点云特征: {len(pointcloud_windows)} 个时间窗口（窗口大小: {window_size}秒）")
    
    # 以点云窗口为基准，搜索图像和其他点云特征
    aligned_image_features = []
    aligned_livox_features = []
    aligned_lidar360_features = []
    aligned_radar_features = []
    aligned_metadata = []
    
    matched_image_count = 0
    unmatched_image_count = 0
    
    for window_start, window_end, window_pointclouds in pointcloud_windows:
        window_center = (window_start + window_end) / 2.0
        
        # 聚合窗口内的点云特征
        livox_feat_in_window = []
        lidar360_feat_in_window = []
        radar_feat_in_window = []
        
        for modality, pc_ts, pc_idx, pc_meta in window_pointclouds:
            if modality == 'livox':
                livox_feat_in_window.append(livox_features[pc_idx])
            elif modality == 'lidar360' and lidar360_features is not None:
                lidar360_feat_in_window.append(lidar360_features[pc_idx])
            elif modality == 'radar' and radar_features is not None:
                radar_feat_in_window.append(radar_features[pc_idx])
        
        # 聚合点云特征（平均池化）
        if livox_feat_in_window:
            livox_feat_agg = torch.stack(livox_feat_in_window).mean(dim=0)
        else:
            livox_feat_agg = torch.zeros(livox_features.shape[1]) if len(livox_features) > 0 else torch.zeros(512)
        
        if lidar360_features is not None:
            if lidar360_feat_in_window:
                lidar360_feat_agg = torch.stack(lidar360_feat_in_window).mean(dim=0)
            else:
                lidar360_feat_agg = torch.zeros(lidar360_features.shape[1])
        else:
            lidar360_feat_agg = None
        
        if radar_features is not None:
            if radar_feat_in_window:
                radar_feat_agg = torch.stack(radar_feat_in_window).mean(dim=0)
            else:
                radar_feat_agg = torch.zeros(radar_features.shape[1])
        else:
            radar_feat_agg = None
        
        # 在窗口内搜索图像特征
        best_image_idx = None
        best_time_diff = float('inf')
        best_image_timestamp = None
        
        for img_ts, img_idx, img_meta in image_timestamps:
            if window_start <= img_ts <= window_end:
                time_diff = abs(img_ts - window_center)
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_image_idx = img_idx
                    best_image_timestamp = img_ts
        
        # 如果找到图像特征，使用；否则使用零向量
        if best_image_idx is not None:
            aligned_image_features.append(image_features[best_image_idx])
            matched_image_count += 1
        else:
            aligned_image_features.append(torch.zeros(image_features.shape[1]) if len(image_features) > 0 else torch.zeros(768))
            unmatched_image_count += 1
        
        aligned_livox_features.append(livox_feat_agg)
        
        if lidar360_feat_agg is not None:
            aligned_lidar360_features.append(lidar360_feat_agg)
        
        if radar_feat_agg is not None:
            aligned_radar_features.append(radar_feat_agg)
        
        # 记录元数据
        aligned_metadata.append({
            'window_start': window_start,
            'window_end': window_end,
            'window_center': window_center,
            'image_timestamp': best_image_timestamp,
            'image_mask': 1.0 if best_image_idx is not None else 0.0,
            'livox_mask': 1.0 if livox_feat_in_window else 0.0,
            'lidar360_mask': 1.0 if lidar360_feat_in_window else 0.0 if lidar360_features is not None else 0.0,
            'radar_mask': 1.0 if radar_feat_in_window else 0.0 if radar_features is not None else 0.0,
            'time_diff': best_time_diff if best_image_idx is not None else None,
            'pointcloud_idx': len(window_pointclouds),
        })
    
    print(f"特征对齐完成:")
    print(f"  点云窗口数: {len(pointcloud_windows)} 个")
    print(f"  匹配到图像: {matched_image_count} 个窗口")
    print(f"  未匹配图像: {unmatched_image_count} 个窗口（使用零向量）")
    
    # 保存对齐后的特征
    aligned_image_tensor = torch.stack(aligned_image_features)
    aligned_livox_tensor = torch.stack(aligned_livox_features)
    
    torch.save(
        {
            "features": aligned_image_tensor,
            "metadata": aligned_metadata,
            "model": "convnext_tiny",
        },
        output_image_path,
    )
    
    torch.save(
        {
            "features": aligned_livox_tensor,
            "metadata": aligned_metadata,
            "model": "pointnext",
        },
        output_livox_path,
    )
    
    if lidar360_features is not None and output_lidar360_path:
        aligned_lidar360_tensor = torch.stack(aligned_lidar360_features)
        torch.save(
            {
                "features": aligned_lidar360_tensor,
                "metadata": aligned_metadata,
                "model": "pointnext",
            },
            output_lidar360_path,
        )
        print(f"  Lidar 360特征: {output_lidar360_path}")
    
    if radar_features is not None and output_radar_path:
        aligned_radar_tensor = torch.stack(aligned_radar_features)
        torch.save(
            {
                "features": aligned_radar_tensor,
                "metadata": aligned_metadata,
                "model": "pointnext",
            },
            output_radar_path,
        )
        print(f"  Radar特征: {output_radar_path}")
    
    print(f"对齐后的特征已保存:")
    print(f"  图像特征: {output_image_path}")
    print(f"  Livox特征: {output_livox_path}")
    
    return True


def align_features_multimodal(
    image_features_path: Path,
    livox_features_path: Path,
    lidar360_features_path: Optional[Path],
    radar_features_path: Optional[Path],
    image_metadata_csv: Optional[Path],
    livox_metadata_csv: Optional[Path],
    lidar360_metadata_csv: Optional[Path],
    radar_metadata_csv: Optional[Path],
    output_image_path: Path,
    output_livox_path: Path,
    output_lidar360_path: Optional[Path],
    output_radar_path: Optional[Path],
    window_size: float = 0.4,
    step_size: float = 0.2,  # 时间窗口滑动步长
):
    """
    三模态独立检测，时间窗口内对齐特征（无主模态设计）。
    
    新策略：
    1. 三种模态（图像、Livox、Lidar 360）独立检测
    2. 构建时间窗口（基于所有检测结果的时间戳范围）
    3. 在每个时间窗口内，搜索三种模态的检测结果
    4. 如果某个模态在窗口内缺失，使用零向量+缺失掩码标记
    5. 保存置信度信息用于加权
    
    参数:
        image_features_path: 图像特征文件路径
        livox_features_path: Livox点云特征文件路径
        lidar360_features_path: Lidar 360点云特征文件路径（可选）
        image_metadata_csv: 图像检测元数据CSV文件路径（可选，用于提取置信度）
        livox_metadata_csv: Livox检测元数据CSV文件路径（可选）
        lidar360_metadata_csv: Lidar 360检测元数据CSV文件路径（可选）
        output_image_path: 对齐后的图像特征输出路径
        output_livox_path: 对齐后的Livox特征输出路径
        output_lidar360_path: 对齐后的Lidar 360特征输出路径（可选）
        window_size: 时间窗口大小（秒，默认0.4秒）
        step_size: 时间窗口滑动步长（秒，默认0.2秒）
    """
    print("=" * 60)
    print("步骤 6: 三模态特征对齐（无主模态，时间窗口内对齐）")
    print("=" * 60)
    
    # 读取三种模态的特征
    image_data = torch.load(image_features_path, map_location='cpu')
    image_features = torch.tensor(image_data['features'])
    image_metadata_list = image_data.get('metadata', [])
    print(f"读取图像特征: {len(image_features)} 个特征")
    
    livox_data = torch.load(livox_features_path, map_location='cpu')
    livox_features = livox_data['features']
    livox_metadata_list = livox_data.get('metadata', [])
    print(f"读取Livox点云特征: {len(livox_features)} 个特征")
    
    lidar360_features = None
    lidar360_metadata_list = []
    if lidar360_features_path and lidar360_features_path.exists():
        lidar360_data = torch.load(lidar360_features_path, map_location='cpu')
        lidar360_features = lidar360_data['features']
        lidar360_metadata_list = lidar360_data.get('metadata', [])
        print(f"读取Lidar 360点云特征: {len(lidar360_features)} 个特征")
    else:
        print("跳过Lidar 360特征（文件不存在或未提供）")
    
    radar_features = None
    radar_metadata_list = []
    if radar_features_path and radar_features_path.exists():
        radar_data = torch.load(radar_features_path, map_location='cpu')
        radar_features = radar_data['features']
        radar_metadata_list = radar_data.get('metadata', [])
        print(f"读取Radar点云特征: {len(radar_features)} 个特征")
    else:
        print("跳过Radar特征（文件不存在或未提供）")
    
    # 构建时间戳索引（从特征元数据中提取）
    image_timestamps = []  # (timestamp, feature_idx, metadata)
    for idx, meta in enumerate(image_metadata_list):
        timestamp = meta.get('timestamp')
        if timestamp is not None:
            image_timestamps.append((float(timestamp), idx, meta))
    
    livox_timestamps = []  # (timestamp, feature_idx, metadata)
    for idx, meta in enumerate(livox_metadata_list):
        timestamp = meta.get('timestamp') or meta.get('timestamp_start')
        if timestamp is not None:
            livox_timestamps.append((float(timestamp), idx, meta))
    
    lidar360_timestamps = []  # (timestamp, feature_idx, metadata)
    if lidar360_features is not None:
        for idx, meta in enumerate(lidar360_metadata_list):
            timestamp = meta.get('timestamp') or meta.get('timestamp_start')
            if timestamp is not None:
                lidar360_timestamps.append((float(timestamp), idx, meta))
    
    radar_timestamps = []  # (timestamp, feature_idx, metadata)
    if radar_features is not None:
        for idx, meta in enumerate(radar_metadata_list):
            timestamp = meta.get('timestamp') or meta.get('timestamp_start')
            if timestamp is not None:
                radar_timestamps.append((float(timestamp), idx, meta))
    
    # 从CSV元数据读取置信度和点云数量（如果提供）
    image_conf_map = {}  # (seq_name, timestamp) -> confidence
    if image_metadata_csv and image_metadata_csv.exists():
        try:
            df = pd.read_csv(image_metadata_csv)
            for _, row in df.iterrows():
                seq_name = row.get('sequence_name', '')
                timestamp = row.get('timestamp', row.get('timestamp_start', 0))
                conf = row.get('confidence', row.get('score', 1.0))
                image_conf_map[(seq_name, float(timestamp))] = float(conf)
        except Exception as e:
            print(f"警告: 读取图像置信度失败: {e}")
    
    livox_conf_map = {}
    livox_point_count_map = {}  # (seq_name, timestamp) -> point_count
    if livox_metadata_csv and livox_metadata_csv.exists():
        try:
            df = pd.read_csv(livox_metadata_csv)
            for _, row in df.iterrows():
                seq_name = row.get('sequence_name', '')
                timestamp = row.get('timestamp_start', 0)
                conf = row.get('score', 1.0)
                point_count = row.get('point_count', 0)  # 点云数量
                livox_conf_map[(seq_name, float(timestamp))] = float(conf)
                livox_point_count_map[(seq_name, float(timestamp))] = int(point_count) if pd.notna(point_count) else 0
        except Exception as e:
            print(f"警告: 读取Livox置信度失败: {e}")
    
    lidar360_conf_map = {}
    lidar360_point_count_map = {}  # (seq_name, timestamp) -> point_count
    if lidar360_metadata_csv and lidar360_metadata_csv.exists():
        try:
            df = pd.read_csv(lidar360_metadata_csv)
            for _, row in df.iterrows():
                seq_name = row.get('sequence_name', '')
                timestamp = row.get('timestamp_start', 0)
                conf = row.get('score', 1.0)
                point_count = row.get('point_count', 0)  # 点云数量
                lidar360_conf_map[(seq_name, float(timestamp))] = float(conf)
                lidar360_point_count_map[(seq_name, float(timestamp))] = int(point_count) if pd.notna(point_count) else 0
        except Exception as e:
            print(f"警告: 读取Lidar 360置信度失败: {e}")
    
    radar_conf_map = {}
    radar_point_count_map = {}  # (seq_name, timestamp) -> point_count
    if radar_metadata_csv and radar_metadata_csv.exists():
        try:
            df = pd.read_csv(radar_metadata_csv)
            for _, row in df.iterrows():
                seq_name = row.get('sequence_name', '')
                timestamp = row.get('timestamp_start', 0)
                conf = row.get('score', 1.0)
                point_count = row.get('point_count', 0)  # 点云数量（从points_path加载的点云文件）
                radar_conf_map[(seq_name, float(timestamp))] = float(conf)
                # 如果CSV中没有point_count，尝试从点云文件加载
                if pd.isna(point_count) or point_count == 0:
                    points_path = row.get('points_path', '')
                    if points_path:
                        try:
                            points_file = Path(points_path)
                            if not points_file.is_absolute():
                                # 尝试在常见的数据根目录下查找
                                for possible_root in [Path('/home/p/MMUAV/data'), Path('/home/p/MMUAV/train')]:
                                    alt_path = possible_root / points_path
                                    if alt_path.exists():
                                        points_file = alt_path
                                        break
                            if points_file.exists():
                                points = np.load(points_file)
                                point_count = len(points) if points.ndim > 0 else 0
                        except:
                            point_count = 0
                radar_point_count_map[(seq_name, float(timestamp))] = int(point_count) if pd.notna(point_count) else 0
        except Exception as e:
            print(f"警告: 读取Radar置信度失败: {e}")
    
    # 找到所有时间戳的范围（添加验证和过滤异常值）
    all_timestamps = []
    
    # 时间戳合理性检查：正常时间戳应该在合理范围内（例如 1700000000.0 ~ 2000000000.0）
    # 如果时间戳看起来像是小数点丢失（例如 170625763540 应该是 1706257635.40），进行修正
    def validate_timestamp(ts: float) -> float:
        """验证并修正时间戳"""
        # 检查时间戳是否异常大（可能是小数点丢失）
        # 正常时间戳通常在 1700000000.0 ~ 2000000000.0 范围内
        if ts > 2000000000.0:
            # 可能是小数点丢失，尝试除以100或1000来修正
            # 例如 170625763540 可能是 1706257635.40（除以100）
            # 或者 1706257635400 可能是 1706257635.400（除以1000）
            original_ts = ts
            if ts > 100000000000.0:  # 超过1000亿，可能是除以1000
                ts = ts / 1000.0
            elif ts > 10000000000.0:  # 超过100亿，可能是除以100
                ts = ts / 100.0
            elif ts > 1000000000.0:  # 超过10亿，可能是除以10
                ts = ts / 10.0
            
            if ts < 2000000000.0:  # 修正后看起来合理
                print(f"警告: 发现异常时间戳 {original_ts:.3f}，已修正为 {ts:.3f}")
                return ts
            else:
                print(f"警告: 发现异常时间戳 {original_ts:.3f}，无法自动修正，将跳过")
                return None
        return ts
    
    # 验证并更新图像时间戳
    if image_timestamps:
        original_count = len(image_timestamps)
        valid_image_ts = []
        updated_image_timestamps = []
        for ts, idx, meta in image_timestamps:
            validated_ts = validate_timestamp(ts)
            if validated_ts is not None:
                valid_image_ts.append(validated_ts)
                updated_image_timestamps.append((validated_ts, idx, meta))
            else:
                print(f"警告: 跳过无效的图像时间戳: {ts} (索引: {idx})")
        image_timestamps[:] = updated_image_timestamps  # 原地更新列表
        all_timestamps.extend(valid_image_ts)
        if len(valid_image_ts) < original_count:
            print(f"警告: 图像时间戳中过滤了 {original_count - len(valid_image_ts)} 个异常值")
    
    # 验证并更新Livox时间戳
    if livox_timestamps:
        original_count = len(livox_timestamps)
        valid_livox_ts = []
        updated_livox_timestamps = []
        for ts, idx, meta in livox_timestamps:
            validated_ts = validate_timestamp(ts)
            if validated_ts is not None:
                valid_livox_ts.append(validated_ts)
                updated_livox_timestamps.append((validated_ts, idx, meta))
            else:
                print(f"警告: 跳过无效的Livox时间戳: {ts} (索引: {idx})")
        livox_timestamps[:] = updated_livox_timestamps  # 原地更新列表
        all_timestamps.extend(valid_livox_ts)
        if len(valid_livox_ts) < original_count:
            print(f"警告: Livox时间戳中过滤了 {original_count - len(valid_livox_ts)} 个异常值")
    
    # 验证并更新Lidar 360时间戳
    if lidar360_timestamps:
        original_count = len(lidar360_timestamps)
        valid_lidar360_ts = []
        updated_lidar360_timestamps = []
        for ts, idx, meta in lidar360_timestamps:
            validated_ts = validate_timestamp(ts)
            if validated_ts is not None:
                valid_lidar360_ts.append(validated_ts)
                updated_lidar360_timestamps.append((validated_ts, idx, meta))
            else:
                print(f"警告: 跳过无效的Lidar 360时间戳: {ts} (索引: {idx})")
        lidar360_timestamps[:] = updated_lidar360_timestamps  # 原地更新列表
        all_timestamps.extend(valid_lidar360_ts)
        if len(valid_lidar360_ts) < original_count:
            print(f"警告: Lidar 360时间戳中过滤了 {original_count - len(valid_lidar360_ts)} 个异常值")
    
    # 验证并更新Radar时间戳
    if radar_timestamps:
        original_count = len(radar_timestamps)
        valid_radar_ts = []
        updated_radar_timestamps = []
        for ts, idx, meta in radar_timestamps:
            validated_ts = validate_timestamp(ts)
            if validated_ts is not None:
                valid_radar_ts.append(validated_ts)
                updated_radar_timestamps.append((validated_ts, idx, meta))
            else:
                print(f"警告: 跳过无效的Radar时间戳: {ts} (索引: {idx})")
        radar_timestamps[:] = updated_radar_timestamps  # 原地更新列表
        all_timestamps.extend(valid_radar_ts)
        if len(valid_radar_ts) < original_count:
            print(f"警告: Radar时间戳中过滤了 {original_count - len(valid_radar_ts)} 个异常值")
    
    if not all_timestamps:
        print("错误: 没有找到任何有效时间戳")
        return False
    
    min_timestamp = min(all_timestamps)
    max_timestamp = max(all_timestamps)
    duration = max_timestamp - min_timestamp
    
    # 检查总时长是否合理（例如不应该超过24小时 = 86400秒）
    if duration > 86400:
        print(f"⚠️  警告: 总时长 {duration:.3f}秒 ({duration/3600:.2f}小时) 看起来异常大，请检查时间戳数据")
    
    print(f"时间戳范围: {min_timestamp:.3f} ~ {max_timestamp:.3f} (总时长: {duration:.3f}秒, {duration/60:.2f}分钟)")
    
    # 构建时间窗口（滑动窗口）
    time_windows = []
    window_start = min_timestamp
    while window_start < max_timestamp:
        window_end = window_start + window_size
        time_windows.append((window_start, window_end))
        window_start += step_size
    
    print(f"构建时间窗口: {len(time_windows)} 个窗口（窗口大小: {window_size}秒，步长: {step_size}秒）")
    
    # 在每个时间窗口内对齐四种模态
    aligned_image_features = []
    aligned_livox_features = []
    aligned_lidar360_features = []
    aligned_radar_features = []
    aligned_metadata = []
    
    for window_start, window_end in time_windows:
        window_center = (window_start + window_end) / 2.0
        
        # 在窗口内搜索图像特征
        image_feat_in_window = []
        image_confs_in_window = []
        for img_ts, img_idx, img_meta in image_timestamps:
            if window_start <= img_ts <= window_end:
                image_feat_in_window.append(image_features[img_idx])
                # 获取置信度
                seq_name = img_meta.get('seq_name', img_meta.get('sequence_name', ''))
                conf = image_conf_map.get((seq_name, img_ts), img_meta.get('confidence', 1.0))
                image_confs_in_window.append(float(conf))
        
        # 在窗口内搜索Livox特征
        livox_feat_in_window = []
        livox_confs_in_window = []
        livox_point_counts_in_window = []
        for livox_ts, livox_idx, livox_meta in livox_timestamps:
            if window_start <= livox_ts <= window_end:
                livox_feat_in_window.append(livox_features[livox_idx])
                seq_name = livox_meta.get('seq_name', livox_meta.get('sequence_name', ''))
                conf = livox_conf_map.get((seq_name, livox_ts), livox_meta.get('score', 1.0))
                point_count = livox_point_count_map.get((seq_name, livox_ts), livox_meta.get('point_count', 0))
                livox_confs_in_window.append(float(conf))
                livox_point_counts_in_window.append(int(point_count) if pd.notna(point_count) else 0)
        
        # 在窗口内搜索Lidar 360特征
        lidar360_feat_in_window = []
        lidar360_confs_in_window = []
        lidar360_point_counts_in_window = []
        if lidar360_features is not None:
            for lidar360_ts, lidar360_idx, lidar360_meta in lidar360_timestamps:
                if window_start <= lidar360_ts <= window_end:
                    lidar360_feat_in_window.append(lidar360_features[lidar360_idx])
                    seq_name = lidar360_meta.get('seq_name', lidar360_meta.get('sequence_name', ''))
                    conf = lidar360_conf_map.get((seq_name, lidar360_ts), lidar360_meta.get('score', 1.0))
                    point_count = lidar360_point_count_map.get((seq_name, lidar360_ts), lidar360_meta.get('point_count', 0))
                    lidar360_confs_in_window.append(float(conf))
                    lidar360_point_counts_in_window.append(int(point_count))
        
        # 在窗口内搜索Radar特征
        radar_feat_in_window = []
        radar_confs_in_window = []
        radar_point_counts_in_window = []
        if radar_features is not None:
            for radar_ts, radar_idx, radar_meta in radar_timestamps:
                if window_start <= radar_ts <= window_end:
                    radar_feat_in_window.append(radar_features[radar_idx])
                    seq_name = radar_meta.get('seq_name', radar_meta.get('sequence_name', ''))
                    conf = radar_conf_map.get((seq_name, radar_ts), radar_meta.get('score', 1.0))
                    point_count = radar_point_count_map.get((seq_name, radar_ts), radar_meta.get('point_count', 0))
                    radar_confs_in_window.append(float(conf))
                    radar_point_counts_in_window.append(int(point_count))
        
        # 聚合窗口内的特征（使用平均池化）
        if image_feat_in_window:
            image_feat_agg = torch.stack(image_feat_in_window).mean(dim=0)
            image_conf_agg = np.mean(image_confs_in_window) if image_confs_in_window else 1.0
            image_mask = 1.0
        else:
            image_feat_agg = torch.zeros(image_features.shape[1])  # 零向量
            image_conf_agg = 0.0
            image_mask = 0.0
        
        if livox_feat_in_window:
            livox_feat_agg = torch.stack(livox_feat_in_window).mean(dim=0)
            livox_conf_agg = np.mean(livox_confs_in_window) if livox_confs_in_window else 1.0
            livox_point_count_agg = np.sum(livox_point_counts_in_window) if livox_point_counts_in_window else 0
            livox_mask = 1.0
        else:
            livox_feat_agg = torch.zeros(livox_features.shape[1])  # 零向量
            livox_conf_agg = 0.0
            livox_point_count_agg = 0
            livox_mask = 0.0
        
        if lidar360_features is not None:
            if lidar360_feat_in_window:
                lidar360_feat_agg = torch.stack(lidar360_feat_in_window).mean(dim=0)
                lidar360_conf_agg = np.mean(lidar360_confs_in_window) if lidar360_confs_in_window else 1.0
                lidar360_point_count_agg = np.sum(lidar360_point_counts_in_window) if lidar360_point_counts_in_window else 0
                lidar360_mask = 1.0
            else:
                lidar360_feat_agg = torch.zeros(lidar360_features.shape[1])  # 零向量
                lidar360_conf_agg = 0.0
                lidar360_point_count_agg = 0
                lidar360_mask = 0.0
        else:
            lidar360_feat_agg = None
            lidar360_conf_agg = 0.0
            lidar360_point_count_agg = 0
            lidar360_mask = 0.0
        
        # 聚合Radar特征
        if radar_features is not None:
            if radar_feat_in_window:
                radar_feat_agg = torch.stack(radar_feat_in_window).mean(dim=0)
                radar_conf_agg = np.mean(radar_confs_in_window) if radar_confs_in_window else 1.0
                radar_point_count_agg = np.sum(radar_point_counts_in_window) if radar_point_counts_in_window else 0
                radar_mask = 1.0
            else:
                radar_feat_agg = torch.zeros(radar_features.shape[1])  # 零向量
                radar_conf_agg = 0.0
                radar_point_count_agg = 0
                radar_mask = 0.0
        else:
            radar_feat_agg = None
            radar_conf_agg = 0.0
            radar_point_count_agg = 0
            radar_mask = 0.0
        
        # 保存对齐后的特征
        aligned_image_features.append(image_feat_agg)
        aligned_livox_features.append(livox_feat_agg)
        if lidar360_feat_agg is not None:
            aligned_lidar360_features.append(lidar360_feat_agg)
        if radar_feat_agg is not None:
            aligned_radar_features.append(radar_feat_agg)
        
        # 保存元数据（包括缺失掩码、置信度和点云数量）
        aligned_metadata.append({
            'window_start': window_start,
            'window_end': window_end,
            'window_center': window_center,
            'image_mask': image_mask,
            'livox_mask': livox_mask,
            'lidar360_mask': lidar360_mask,
            'radar_mask': radar_mask,
            'image_confidence': image_conf_agg,
            'livox_confidence': livox_conf_agg,
            'lidar360_confidence': lidar360_conf_agg,
            'radar_confidence': radar_conf_agg,
            'image_point_count': len(image_feat_in_window),  # 图像使用检测框数量
            'livox_point_count': livox_point_count_agg,
            'lidar360_point_count': lidar360_point_count_agg,
            'radar_point_count': radar_point_count_agg,
            'image_count': len(image_feat_in_window),
            'livox_count': len(livox_feat_in_window),
            'lidar360_count': len(lidar360_feat_in_window),
            'radar_count': len(radar_feat_in_window),
        })
    
    print(f"特征对齐完成:")
    print(f"  时间窗口数: {len(time_windows)}")
    print(f"  图像窗口有特征: {sum(1 for m in aligned_metadata if m['image_mask'] > 0)}")
    print(f"  Livox窗口有特征: {sum(1 for m in aligned_metadata if m['livox_mask'] > 0)}")
    if lidar360_features is not None:
        print(f"  Lidar 360窗口有特征: {sum(1 for m in aligned_metadata if m['lidar360_mask'] > 0)}")
    if radar_features is not None:
        print(f"  Radar窗口有特征: {sum(1 for m in aligned_metadata if m.get('radar_mask', 0) > 0)}")
    
    # 保存对齐后的特征
    aligned_image_tensor = torch.stack(aligned_image_features)
    aligned_livox_tensor = torch.stack(aligned_livox_features)
    
    torch.save(
        {
            "features": aligned_image_tensor,
            "metadata": aligned_metadata,
            "model": "convnext_tiny",
        },
        output_image_path,
    )
    
    torch.save(
        {
            "features": aligned_livox_tensor,
            "metadata": aligned_metadata,
            "model": "pointnext",
        },
        output_livox_path,
    )
    
    if lidar360_features is not None and output_lidar360_path:
        aligned_lidar360_tensor = torch.stack(aligned_lidar360_features)
        torch.save(
            {
                "features": aligned_lidar360_tensor,
                "metadata": aligned_metadata,
                "model": "pointnext",
            },
            output_lidar360_path,
        )
        print(f"  Lidar 360特征: {output_lidar360_path}")
    
    if radar_features is not None and output_radar_path:
        aligned_radar_tensor = torch.stack(aligned_radar_features)
        torch.save(
            {
                "features": aligned_radar_tensor,
                "metadata": aligned_metadata,
                "model": "pointnext",
            },
            output_radar_path,
        )
        print(f"  Radar特征: {output_radar_path}")
    
    print(f"对齐后的特征已保存:")
    print(f"  图像特征: {output_image_path}")
    print(f"  Livox特征: {output_livox_path}")
    if lidar360_features is not None:
        print(f"  Lidar 360特征: {output_lidar360_path}")
    if radar_features is not None:
        print(f"  Radar特征: {output_radar_path}")
    
    return True


def main():
    args = parse_args()
    
    # 如果指定了 --window-sizes，运行实验模式
    if args.window_sizes:
        import copy
        
        print("=" * 80)
        print("时间窗口大小分析实验")
        print("=" * 80)
        print(f"窗口大小列表: {args.window_sizes}")
        if args.experiment_mode:
            print("实验模式: 为每个窗口大小创建独立的特征目录")
        print()
        
        results = {}
        base_convnext_output = args.convnext_output
        base_pointnext_output = args.pointnext_output
        base_lidar360_output = args.lidar360_pointnext_output
        base_radar_output = args.radar_pointnext_output
        
        for window_size in args.window_sizes:
            print("\n" + "=" * 80)
            print(f"处理窗口大小: {window_size}s")
            print("=" * 80)
            
            # 创建新的参数对象（浅拷贝）
            args.window_size = window_size
            
            # 实验模式：为每个窗口大小创建独立的特征目录
            if args.experiment_mode:
                base_features_dir = Path(base_convnext_output).parent
                window_features_dir = base_features_dir / f"features_window_{window_size}"
                window_features_dir.mkdir(parents=True, exist_ok=True)
                
                # 更新输出路径
                args.convnext_output = str(window_features_dir / "image_features")
                args.pointnext_output = str(window_features_dir / "livox_features")
                if base_lidar360_output:
                    args.lidar360_pointnext_output = str(window_features_dir / "lidar360_features")
                if base_radar_output:
                    args.radar_pointnext_output = str(window_features_dir / "radar_features")
                
                print(f"特征输出目录: {window_features_dir}")
            
            try:
                # 运行主流程（继续执行后续代码）
                _run_main_process(args)
                results[window_size] = "成功"
                print(f"\n✓ 窗口大小 {window_size}s 处理完成")
            except Exception as e:
                results[window_size] = f"失败: {str(e)}"
                print(f"\n✗ 窗口大小 {window_size}s 处理失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 恢复基础输出路径（为下次循环准备）
            if args.experiment_mode:
                args.convnext_output = base_convnext_output
                args.pointnext_output = base_pointnext_output
                args.lidar360_pointnext_output = base_lidar360_output
                args.radar_pointnext_output = base_radar_output
        
        # 打印总结
        print("\n" + "=" * 80)
        print("时间窗口大小分析实验总结")
        print("=" * 80)
        for window_size, status in results.items():
            print(f"  窗口大小 {window_size}s: {status}")
        print("=" * 80)
        return
    
    # 正常模式：单个窗口大小
    _run_main_process(args)


def _run_main_process(args):
    """运行主要的特征提取和对齐流程（从原来的 main 函数提取）"""
    # 如果只处理一个序列，自动启用按序列分别保存（避免覆盖）
    if args.sequences and len(args.sequences) == 1:
        if not args.per_sequence_output:
            print(f"检测到只处理一个序列 ({args.sequences[0]})，自动启用 --per-sequence-output 以避免覆盖")
            args.per_sequence_output = True
    
    # 如果启用按序列分别保存，调整输出路径
    if args.per_sequence_output and args.sequences:
        # 检测数据分割（train/val）
        data_root_path = Path(args.data_root)
        if data_root_path.name in ["train", "val", "test"]:
            data_split = data_root_path.name
        elif (data_root_path / "train").exists():
            data_split = "train"
        elif (data_root_path / "val").exists():
            data_split = "val"
        elif (data_root_path / "test").exists():
            data_split = "test"
        else:
            data_split = "train"
        
        # 为每个序列分别设置输出路径（只处理第一个序列，因为每次只处理一个）
        if len(args.sequences) == 1:
            seq_name = args.sequences[0]
            seq_suffix = f"_{seq_name}"
            
            def add_seq_suffix_to_path(path_str, default_name):
                """为路径添加序列后缀"""
                if not path_str:
                    return path_str
                path_obj = Path(path_str)
                if data_split in path_obj.stem:
                    # 如果路径中已有split名称，添加序列部分
                    new_stem = path_obj.stem.replace(f'_{data_split}', seq_suffix + f'_{data_split}')
                else:
                    new_stem = f"{path_obj.stem}{seq_suffix}"
                return str(path_obj.parent / f"{new_stem}.pt")
            
            # 更新点云特征输出路径（Livox）
            args.pointnext_output = add_seq_suffix_to_path(args.pointnext_output, "livox_features")
            
            # 更新图像特征输出路径
            args.convnext_output = add_seq_suffix_to_path(args.convnext_output, "image_features")
            
            # 更新Lidar 360特征输出路径
            if args.lidar360_pointnext_output:
                args.lidar360_pointnext_output = add_seq_suffix_to_path(args.lidar360_pointnext_output, "lidar360_features")
            
            # 更新Radar特征输出路径
            if args.radar_pointnext_output:
                args.radar_pointnext_output = add_seq_suffix_to_path(args.radar_pointnext_output, "radar_features")
            
            print(f"按序列分别保存特征文件: {seq_name}")
            print(f"  点云特征 (Livox): {args.pointnext_output}")
            print(f"  图像特征: {args.convnext_output}")
            if args.lidar360_pointnext_output:
                print(f"  点云特征 (Lidar 360): {args.lidar360_pointnext_output}")
            if args.radar_pointnext_output:
                print(f"  点云特征 (Radar): {args.radar_pointnext_output}")
        else:
            print("警告: --per-sequence-output 模式下，建议每次只处理一个序列")
    
    # 创建输出目录
    Path(args.pointnext_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.convnext_output).parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("多模态特征提取和对齐流程")
    print("=" * 60)
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"对齐策略: {args.alignment_strategy}")
    print(f"特征对齐模式: {args.feature_alignment_mode}")
    print()
    
    # 步骤1: 数据对齐
    run_alignment(args)
    print()
    
    # 步骤2: 图像检测
    run_image_detection(args)
    print()
    
    # 步骤3: 点云检测（Livox Avia）
    run_pointcloud_detection(args)
    print()
    
    # 步骤3.1: Lidar 360 检测
    run_lidar360_detection(args)
    print()
    
    # 步骤3.2: Radar Enhance 检测
    run_radar_detection(args)
    print()
    
    # 步骤4: 提取图像特征（从检测结果）
    convnext_success = run_convnext_extraction(args)
    if not convnext_success:
        print("⚠️  图像特征提取失败，无法继续执行对齐步骤")
        print("请检查特征提取步骤的错误信息，修复问题后重新运行")
        return
    print()
    
    # 步骤5: 提取点云特征（从检测结果）
    # 5.1: Livox Avia
    pointnext_success = run_pointnext_extraction(args)
    if not pointnext_success:
        print("⚠️  Livox点云特征提取失败，无法继续执行对齐步骤")
        print("请检查特征提取步骤的错误信息，修复问题后重新运行")
        return
    print()
    
    # 5.2: Lidar 360
    lidar360_success = run_lidar360_pointnext_extraction(args)
    print()
    
    # 5.3: Radar Enhance
    radar_success = run_radar_pointnext_extraction(args)
    print()
    
    # 验证必要的特征文件是否存在
    print("验证特征文件...")
    convnext_path = Path(args.convnext_output)
    pointnext_path = Path(args.pointnext_output)
    
    if not convnext_path.exists():
        print(f"错误: 图像特征文件不存在: {convnext_path}")
        print(f"请检查特征提取步骤是否成功完成")
        return
    
    if not pointnext_path.exists():
        print(f"错误: Livox点云特征文件不存在: {pointnext_path}")
        print(f"请检查特征提取步骤是否成功完成")
        return
    
    print(f"✓ 图像特征文件存在: {convnext_path}")
    print(f"✓ Livox点云特征文件存在: {pointnext_path}")
    print()
    
    # 步骤6: 四模态特征对齐（新策略：无主模态，时间窗口内对齐）
    data_root_path = Path(args.data_root)
    
    # 查找四种模态的检测元数据文件
    livox_metadata_path = data_root_path / args.livox_metadata_filename
    lidar360_metadata_path = data_root_path / args.lidar360_metadata_filename
    radar_metadata_path = data_root_path / args.radar_metadata_filename
    
    # 图像检测元数据（从YOLO检测结果中提取，如果存在）
    image_metadata_path = None
    # TODO: 可以从YOLO检测结果中提取图像检测元数据
    
    # 创建对齐后的特征文件路径（如果使用per-sequence-output，文件名包含序列名）
    convnext_base = Path(args.convnext_output)
    pointnext_base = Path(args.pointnext_output)
    lidar360_base = Path(args.lidar360_pointnext_output) if args.lidar360_pointnext_output else None
    radar_base = Path(args.radar_pointnext_output) if args.radar_pointnext_output else None
    
    # 从输入路径提取序列名（如果存在）
    seq_suffix = ""
    if args.per_sequence_output and args.sequences and len(args.sequences) == 1:
        seq_suffix = f"_{args.sequences[0]}"
    
    aligned_image_path = convnext_base.parent / f"image_features_aligned{seq_suffix}.pt"
    aligned_livox_path = pointnext_base.parent / f"livox_features_aligned{seq_suffix}.pt"
    aligned_lidar360_path = None
    if lidar360_base:
        aligned_lidar360_path = lidar360_base.parent / f"lidar360_features_aligned{seq_suffix}.pt"
    aligned_radar_path = None
    if radar_base:
        aligned_radar_path = radar_base.parent / f"radar_features_aligned{seq_suffix}.pt"
    
    # 根据特征对齐模式选择不同的对齐策略
    if args.feature_alignment_mode == "multimodal":
        print("执行四模态特征对齐（无主模态，时间窗口内对齐）...")
        success = align_features_multimodal(
            image_features_path=Path(args.convnext_output),
            livox_features_path=Path(args.pointnext_output),
            lidar360_features_path=Path(args.lidar360_pointnext_output) if args.lidar360_pointnext_output else None,
            radar_features_path=Path(args.radar_pointnext_output) if args.radar_pointnext_output else None,
            image_metadata_csv=image_metadata_path,
            livox_metadata_csv=livox_metadata_path if livox_metadata_path.exists() else None,
            lidar360_metadata_csv=lidar360_metadata_path if lidar360_metadata_path.exists() else None,
            radar_metadata_csv=radar_metadata_path if radar_metadata_path.exists() else None,
            output_image_path=aligned_image_path,
            output_livox_path=aligned_livox_path,
            output_lidar360_path=aligned_lidar360_path,
            output_radar_path=aligned_radar_path,
            window_size=args.window_size,
            step_size=args.window_size / 2.0  # 步长为窗口大小的一半，实现滑动窗口
        )
    elif args.feature_alignment_mode == "image_led":
        print("执行特征对齐（以图像为主导模态）...")
        # 检查特征文件是否存在
        convnext_path = Path(args.convnext_output)
        pointnext_path = Path(args.pointnext_output)
        
        if not convnext_path.exists():
            print(f"错误: 图像特征文件不存在: {convnext_path}")
            print(f"请确保特征提取步骤已完成，或检查路径是否正确")
            success = False
        elif not pointnext_path.exists():
            print(f"错误: Livox点云特征文件不存在: {pointnext_path}")
            print(f"请确保特征提取步骤已完成，或检查路径是否正确")
            success = False
        else:
            success = align_features_by_image_4modal(
                image_features_path=convnext_path,
                livox_features_path=pointnext_path,
                lidar360_features_path=Path(args.lidar360_pointnext_output) if args.lidar360_pointnext_output and Path(args.lidar360_pointnext_output).exists() else None,
                radar_features_path=Path(args.radar_pointnext_output) if args.radar_pointnext_output and Path(args.radar_pointnext_output).exists() else None,
                image_metadata_csv=image_metadata_path,
                livox_metadata_csv=livox_metadata_path if livox_metadata_path.exists() else None,
                lidar360_metadata_csv=lidar360_metadata_path if lidar360_metadata_path.exists() else None,
                radar_metadata_csv=radar_metadata_path if radar_metadata_path.exists() else None,
                output_image_path=aligned_image_path,
                output_livox_path=aligned_livox_path,
                output_lidar360_path=aligned_lidar360_path,
                output_radar_path=aligned_radar_path,
                window_size=args.window_size,
                max_time_diff=args.window_size / 2.0
            )
    elif args.feature_alignment_mode == "pointcloud_led":
        print("执行特征对齐（以点云为主导模态）...")
        # 检查特征文件是否存在
        convnext_path = Path(args.convnext_output)
        pointnext_path = Path(args.pointnext_output)
        
        if not convnext_path.exists():
            print(f"错误: 图像特征文件不存在: {convnext_path}")
            print(f"请确保特征提取步骤已完成，或检查路径是否正确")
            success = False
        elif not pointnext_path.exists():
            print(f"错误: Livox点云特征文件不存在: {pointnext_path}")
            print(f"请确保特征提取步骤已完成，或检查路径是否正确")
            success = False
        else:
            success = align_features_by_pointcloud(
                image_features_path=convnext_path,
                livox_features_path=pointnext_path,
                lidar360_features_path=Path(args.lidar360_pointnext_output) if args.lidar360_pointnext_output and Path(args.lidar360_pointnext_output).exists() else None,
                radar_features_path=Path(args.radar_pointnext_output) if args.radar_pointnext_output and Path(args.radar_pointnext_output).exists() else None,
                image_metadata_csv=image_metadata_path,
                livox_metadata_csv=livox_metadata_path if livox_metadata_path.exists() else None,
                lidar360_metadata_csv=lidar360_metadata_path if lidar360_metadata_path.exists() else None,
                radar_metadata_csv=radar_metadata_path if radar_metadata_path.exists() else None,
                output_image_path=aligned_image_path,
                output_livox_path=aligned_livox_path,
                output_lidar360_path=aligned_lidar360_path,
                output_radar_path=aligned_radar_path,
                window_size=args.window_size,
                max_time_diff=args.window_size / 2.0
            )
    else:
        print(f"错误: 未知的特征对齐模式: {args.feature_alignment_mode}")
        success = False
    
    if success:
        # 更新输出路径为对齐后的特征
        args.convnext_output = str(aligned_image_path)
        args.pointnext_output = str(aligned_livox_path)
        if aligned_lidar360_path:
            args.lidar360_pointnext_output = str(aligned_lidar360_path)
        if aligned_radar_path:
            args.radar_pointnext_output = str(aligned_radar_path)
        if args.lidar360_pointnext_output and aligned_lidar360_path:
            args.lidar360_pointnext_output = str(aligned_lidar360_path)
        print("✅ 三模态特征对齐完成，使用对齐后的特征文件")
    else:
        print("⚠️  特征对齐失败，使用原始特征文件")
    print()
    
    print("=" * 60)
    print("所有步骤完成！")
    print("=" * 60)
    print(f"对齐结果: {args.output_dir}")
    print(f"点云特征: {args.pointnext_output}")
    print(f"图像特征: {args.convnext_output}")
    print()
    print("下一步: 运行训练脚本")
    print("  python train_multimodal_classifier.py \\")
    print(f"    --pointnext-features {args.pointnext_output} \\")
    print(f"    --convnext-features {args.convnext_output} \\")
    print(f"    --timeline-dir {args.output_dir}")


if __name__ == "__main__":
    main()

