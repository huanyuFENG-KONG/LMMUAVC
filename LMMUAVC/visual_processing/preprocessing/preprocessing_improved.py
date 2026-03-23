#%%
import os
import argparse
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import numpy as np

# 默认配置（可通过命令行参数覆盖）
DEFAULT_ROOT = Path('/home/p/MMUAV/data')
DEFAULT_PREFIX = ''  # 如果数据直接在root下，prefix为空；否则为 'train' 或 'val'
MODALS = ('lidar_360', 'Image', 'livox_avia', 'ground_truth', 'radar_enhance_pcl', 'class')
BASE_MODAL = 'lidar_360'
DEFAULT_MAX_SEQ = 8  # 默认支持8个序列: seq0001 ~ seq0008

# 对齐策略配置
DEFAULT_ALIGNMENT_STRATEGY = 'time_window'  # 'nearest', 'time_window', 'one_to_many', 'downsample'
DEFAULT_MAX_TIME_DIFF = 0.1  # 最大允许时间差（秒），用于时间窗口策略


#%% 函数
def get_seq(seq=1, modal='ground_truth', root=None, prefix='', seq_format='seq{:04d}') -> OrderedDict[float, Path]:
    """
    获取模态的文件名和路径

    当前仅用于get_timestamp_list
    :param seq: Sequence id
    :param modal: One of ['class', 'ground_truth', 'Image', 'lidar_360', 'livox_avia', 'radar_enhance_pcl']
    :param root: Dataset base folder
    :param prefix: 前缀目录（如 'train' 或 'val'），如果数据直接在root下则为空字符串
    :param seq_format: 序列命名格式，默认 'seq{:04d}' (seq0001) 或 'seq{}' (seq1)
    :return: An OrderedDict from timestamp to file path, the order is based on timestamp. For example {'170001.12345': Path('/path/to/modal/170001.12345.npy'), ...}
    """
    if root is None:
        root = DEFAULT_ROOT
    base_path = root / prefix if prefix else root
    seq_name = seq_format.format(seq)
    base_dir = base_path / seq_name / modal
    if not base_dir.exists():
        # 尝试其他格式
        if seq_format == 'seq{:04d}':
            seq_name = f"seq{seq}"
            base_dir = base_path / seq_name / modal
    # 只处理可以转换为浮点数的时间戳文件名，跳过其他文件（如 correction_stats.json）
    pairs = []
    for x in base_dir.iterdir():
        if x.is_file():
            try:
                timestamp = float(x.stem)
                pairs.append((timestamp, x))
            except ValueError:
                # 跳过无法转换为浮点数的文件名（如 correction_stats.json）
                continue
    pairs.sort(key=lambda x: x[0])
    return OrderedDict(pairs)


def get_timestamp_list(seq=1, modal=BASE_MODAL, root=None, prefix='', seq_format='seq{:04d}') -> list[float]:
    return list(get_seq(seq, modal, root, prefix, seq_format).keys())


def get_closet_timestamp(t, serial):
    """找到最近的一个点（原始方法）"""
    a = max([x for x in serial if x < t] or [-float('inf')])
    b = min([x for x in serial if x >= t] or [float('inf')])

    return a if t - a <= b - t else b


def get_timestamps_in_window(t, serial, max_diff):
    """
    找到时间窗口内的所有时间戳
    
    :param t: 目标时间戳
    :param serial: 时间戳序列
    :param max_diff: 最大时间差（秒）
    :return: 时间窗口内的时间戳列表，按距离排序
    """
    candidates = [ts for ts in serial if abs(ts - t) <= max_diff]
    candidates.sort(key=lambda x: abs(x - t))
    return candidates


def get_closest_with_threshold(t, serial, max_diff):
    """
    找到最近的时间戳，但要求时间差在阈值内
    
    :param t: 目标时间戳
    :param serial: 时间戳序列
    :param max_diff: 最大允许时间差（秒）
    :return: 最近的时间戳，如果超出阈值则返回None
    """
    closest = get_closet_timestamp(t, serial)
    if abs(closest - t) <= max_diff:
        return closest
    return None


#%% 对齐策略
def align_seq_nearest(timestamps_dict: dict[str, list[float]]):
    """
    原始对齐方法：一对一最近邻匹配
    
    问题：如果雷达8帧、激光雷达4帧、相机12帧
    - 以激光雷达4帧为基准
    - 每个激光雷达帧只匹配一个最近的雷达帧和一个最近的相机帧
    - 结果：丢失了4个雷达帧和8个相机帧的信息
    """
    result = {'average': [], **{m: [] for m in MODALS}}
    for t in timestamps_dict[BASE_MODAL]:
        closet = {m: get_closet_timestamp(t, timestamps_dict[m]) for m in MODALS}
        average = sum(closet.values()) / len(closet.values())
        result['average'].append(average)
        for m in MODALS:
            result[m].append(closet[m])
    return result


def align_seq_time_window(timestamps_dict: dict[str, list[float]], max_time_diff: float):
    """
    时间窗口对齐方法：在时间窗口内选择最近的时间戳
    
    优点：
    - 可以过滤掉时间差过大的匹配
    - 保证对齐质量
    
    仍然是一对一，但增加了质量检查
    """
    result = {'average': [], **{m: [] for m in MODALS}, 'max_diff': []}
    for t in timestamps_dict[BASE_MODAL]:
        closet = {}
        max_diff = 0
        for m in MODALS:
            closest_ts = get_closest_with_threshold(t, timestamps_dict[m], max_time_diff)
            if closest_ts is not None:
                closet[m] = closest_ts
                max_diff = max(max_diff, abs(closest_ts - t))
            else:
                # 如果找不到在阈值内的，使用最近邻（但记录警告）
                closet[m] = get_closet_timestamp(t, timestamps_dict[m])
                max_diff = max(max_diff, abs(closet[m] - t))
        
        average = sum(closet.values()) / len(closet.values())
        result['average'].append(average)
        result['max_diff'].append(max_diff)
        for m in MODALS:
            result[m].append(closet[m])
    return result


def align_seq_one_to_many(timestamps_dict: dict[str, list[float]], max_time_diff: float):
    """
    一对多对齐方法：一个基准帧可以对应多个高频模态帧
    
    适用于：雷达8帧、激光雷达4帧、相机12帧的情况
    - 以激光雷达4帧为基准
    - 每个激光雷达帧可以匹配多个雷达帧和相机帧
    - 结果：保留所有帧的信息
    
    策略：
    1. 对于每个基准帧，找到时间窗口内所有模态的候选帧
    2. 为每个高频模态的每个候选帧生成一行（如果候选数>1）
    3. 同时保留主匹配行（最接近的帧）
    
    输出格式：每个基准帧可能生成多行，每行对应一个高频模态的候选组合
    """
    result_rows = []
    
    for t_base in timestamps_dict[BASE_MODAL]:
        # 对于基准模态的每个时间戳，找到所有模态在时间窗口内的帧
        modal_candidates = {}
        for m in MODALS:
            if m == BASE_MODAL:
                modal_candidates[m] = [t_base]  # 基准模态只有自己
            else:
                candidates = get_timestamps_in_window(t_base, timestamps_dict[m], max_time_diff)
                if not candidates:
                    # 如果窗口内没有候选，使用最近邻
                    candidates = [get_closet_timestamp(t_base, timestamps_dict[m])]
                modal_candidates[m] = candidates
        
        # 找到每个模态最接近的帧作为主匹配
        main_match = {}
        for m in MODALS:
            if m == BASE_MODAL:
                main_match[m] = t_base
            else:
                # 选择距离基准时间戳最近的候选
                candidates = modal_candidates[m]
                main_match[m] = min(candidates, key=lambda x: abs(x - t_base))
        
        # 生成主匹配行
        average = sum(main_match.values()) / len(main_match.values())
        main_row = {'average': average, 'base_timestamp': t_base}
        for m in MODALS:
            main_row[m] = main_match[m]
        main_row['is_primary'] = True  # 标记为主匹配
        result_rows.append(main_row)
        
        # 为高频模态的额外候选帧生成额外行
        # 策略：如果某个模态有多个候选（>1），为每个额外候选生成一行
        # 其他模态使用主匹配值
        for m in MODALS:
            if m != BASE_MODAL and len(modal_candidates[m]) > 1:
                # 为主匹配之外的每个候选生成一行
                for candidate_ts in modal_candidates[m]:
                    if candidate_ts != main_match[m]:  # 跳过主匹配
                        extra_row = {'average': average, 'base_timestamp': t_base}
                        for other_m in MODALS:
                            if other_m == m:
                                extra_row[other_m] = candidate_ts
                            else:
                                extra_row[other_m] = main_match[other_m]
                        extra_row['is_primary'] = False  # 标记为额外匹配
                        result_rows.append(extra_row)
    
    # 转换为DataFrame格式
    result = {'average': [], **{m: [] for m in MODALS}}
    for row in result_rows:
        result['average'].append(row['average'])
        for m in MODALS:
            result[m].append(row[m])
    
    return result


def align_seq_downsample(timestamps_dict: dict[str, list[float]]):
    """
    降采样对齐方法：将所有模态降采样到最低频率
    
    适用于：需要统一采样率的情况
    - 找到所有模态中采样率最低的（时间间隔最大的）
    - 以该频率为基准，对所有模态进行降采样
    - 结果：所有模态帧数相同，但可能丢失高频信息
    """
    # 计算每个模态的平均采样间隔
    modal_intervals = {}
    for m in MODALS:
        timestamps = timestamps_dict[m]
        if len(timestamps) > 1:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            modal_intervals[m] = np.mean(intervals)
        else:
            modal_intervals[m] = float('inf')
    
    # 找到最大间隔（最低频率）
    max_interval = max(modal_intervals.values())
    target_interval = max_interval
    
    # 生成统一的时间戳序列（从最早到最晚，按目标间隔）
    all_timestamps = []
    for timestamps in timestamps_dict.values():
        all_timestamps.extend(timestamps)
    start_time = min(all_timestamps)
    end_time = max(all_timestamps)
    
    # 生成目标时间戳序列
    target_timestamps = np.arange(start_time, end_time + target_interval, target_interval)
    
    # 对每个目标时间戳，找到各模态最近的时间戳
    result = {'average': [], **{m: [] for m in MODALS}}
    for t in target_timestamps:
        closet = {m: get_closet_timestamp(t, timestamps_dict[m]) for m in MODALS}
        average = sum(closet.values()) / len(closet.values())
        result['average'].append(average)
        for m in MODALS:
            result[m].append(closet[m])
    
    return result


#%% 统一对齐接口
def align_seq(timestamps_dict: dict[str, list[float]], strategy=DEFAULT_ALIGNMENT_STRATEGY, max_time_diff=DEFAULT_MAX_TIME_DIFF):
    """
    对齐一个timestamps_dict，支持多种策略

    :param timestamps_dict: (用于对齐的)多模态文件数据结构
    :param strategy: 对齐策略
        - 'nearest': 最近邻匹配（原始方法）
        - 'time_window': 时间窗口匹配（带质量检查）
        - 'one_to_many': 一对多匹配（保留高频信息）
        - 'downsample': 降采样匹配（统一采样率）
    :param max_time_diff: 最大时间差（秒），用于 time_window 和 one_to_many 策略
    """
    if strategy == 'nearest':
        return align_seq_nearest(timestamps_dict)
    elif strategy == 'time_window':
        return align_seq_time_window(timestamps_dict, max_time_diff)
    elif strategy == 'one_to_many':
        return align_seq_one_to_many(timestamps_dict, max_time_diff)
    elif strategy == 'downsample':
        return align_seq_downsample(timestamps_dict)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


#%% 对齐质量评估
def evaluate_alignment_quality(result_df: pd.DataFrame):
    """
    评估对齐质量
    
    :param result_df: 对齐后的DataFrame
    :return: 质量统计信息
    """
    stats = {}
    
    # 计算每个样本的时间差
    base_timestamps = result_df[BASE_MODAL].values
    for m in MODALS:
        if m != BASE_MODAL:
            diffs = np.abs(result_df[m].values - base_timestamps)
            stats[f'{m}_mean_diff'] = np.mean(diffs)
            stats[f'{m}_max_diff'] = np.max(diffs)
            stats[f'{m}_std_diff'] = np.std(diffs)
    
    # 计算平均时间戳的方差（越小说明对齐越好）
    avg_timestamps = result_df['average'].values
    stats['average_variance'] = np.var(avg_timestamps)
    
    return stats


#%%
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='多模态数据对齐脚本')
    parser.add_argument(
        '--data-root',
        type=str,
        default=str(DEFAULT_ROOT),
        help=f'数据集根目录（默认: {DEFAULT_ROOT}）'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default=DEFAULT_PREFIX,
        help='前缀目录（如 train 或 val），如果数据直接在root下则为空（默认: 空）'
    )
    parser.add_argument(
        '--alignment-strategy',
        type=str,
        default=DEFAULT_ALIGNMENT_STRATEGY,
        choices=['nearest', 'time_window', 'one_to_many', 'downsample'],
        help=f'对齐策略（默认: {DEFAULT_ALIGNMENT_STRATEGY}）'
    )
    parser.add_argument(
        '--max-time-diff',
        type=float,
        default=DEFAULT_MAX_TIME_DIFF,
        help=f'最大时间差（秒），用于时间窗口策略（默认: {DEFAULT_MAX_TIME_DIFF}）'
    )
    parser.add_argument(
        '--max-seq',
        type=int,
        default=DEFAULT_MAX_SEQ,
        help=f'最大序列数（默认: {DEFAULT_MAX_SEQ}）'
    )
    parser.add_argument(
        '--seq-ids',
        type=int,
        nargs='+',
        default=None,
        help='指定要处理的序列ID列表（例如: --seq-ids 2 表示只处理seq0002），如果指定则忽略max-seq'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认: out/{prefix}）'
    )
    parser.add_argument(
        '--seq-format',
        type=str,
        default='seq{:04d}',
        choices=['seq{:04d}', 'seq{}'],
        help='序列命名格式：seq{:04d} (seq0001) 或 seq{} (seq1)（默认: seq{:04d}）'
    )
    return parser.parse_args()


def main(args=None):
    """主函数"""
    if args is None:
        args = parse_args()
    
    root = Path(args.data_root)
    prefix = args.prefix
    alignment_strategy = args.alignment_strategy
    max_time_diff = args.max_time_diff
    max_seq = args.max_seq
    seq_format = args.seq_format
    seq_ids_to_process = args.seq_ids
    
    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('out') / (prefix if prefix else 'data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"数据根目录: {root}")
    print(f"前缀: {prefix if prefix else '(无)'}")
    print(f"对齐策略: {alignment_strategy}")
    print(f"最大时间差: {max_time_diff}s")
    print(f"输出目录: {output_dir}")
    if seq_ids_to_process:
        print(f"指定序列ID: {seq_ids_to_process}")
    print()
    
    # 确定要处理的序列ID列表
    if seq_ids_to_process:
        # 如果指定了seq_ids，只处理这些序列
        seq_ids_to_check = seq_ids_to_process
    else:
        # 否则，从1到max_seq查找存在的序列
        seq_ids_to_check = list(range(1, max_seq + 1))
    
    # 先检查哪些序列目录存在，只处理存在的序列
    existing_seq_ids = []
    for seq_id in seq_ids_to_check:
        seq_name = seq_format.format(seq_id)
        seq_dir = root / prefix / seq_name if prefix else root / seq_name
        if seq_dir.exists() and seq_dir.is_dir():
            existing_seq_ids.append(seq_id)
        elif seq_ids_to_process:
            # 如果用户明确指定了序列但不存在，给出警告
            print(f"警告: 指定的序列 {seq_name} (seq_id={seq_id}) 不存在，跳过")
    
    if not existing_seq_ids:
        print(f"错误: 在 {root}/{prefix if prefix else ''} 下未找到任何序列目录")
        return
    
    print(f"找到 {len(existing_seq_ids)} 个序列目录: {[seq_format.format(s) for s in existing_seq_ids]}\n")
    
    for seq_id in existing_seq_ids:
        seq_name = seq_format.format(seq_id)
        print(f"处理序列: {seq_name} (seq_id={seq_id})")

        try:
            timestamp_set = {m: get_timestamp_list(seq_id, m, root, prefix, seq_format) for m in MODALS}
        except (FileNotFoundError, KeyError) as e:
            print(f"  警告: 序列 {seq_name} 不存在或数据不完整，跳过")
            print(f"  错误: {e}")
            continue
        
        # 检查是否有数据
        if not any(timestamp_set.values()):
            print(f"  警告: 序列 {seq_name} 没有数据，跳过")
            continue
        
        # 打印采样频率信息
        print(f"  采样频率统计:")
        for m in MODALS:
            timestamps = timestamp_set[m]
            if len(timestamps) > 1:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = np.mean(intervals)
                freq = 1.0 / avg_interval if avg_interval > 0 else 0
                print(f"    {m}: {len(timestamps)}帧, 平均间隔={avg_interval:.4f}s, 频率={freq:.2f}Hz")
        
        result = align_seq(timestamp_set, strategy=alignment_strategy, max_time_diff=max_time_diff)

        result_df = pd.DataFrame(result)
        result_df['seq'] = pd.Series(seq_id, index=result_df.index)
        
        # 一对多策略的特殊处理：统计生成的样本数
        if alignment_strategy == 'one_to_many':
            base_frame_count = len(timestamp_set[BASE_MODAL])
            output_frame_count = len(result_df)
            expansion_ratio = output_frame_count / base_frame_count if base_frame_count > 0 else 1.0
            print(f"  一对多对齐结果: 基准帧{base_frame_count}个 -> 输出样本{output_frame_count}个 (扩展比: {expansion_ratio:.2f}x)")
        
        # 评估对齐质量
        quality = evaluate_alignment_quality(result_df)
        max_diff = max([quality.get(f'{m}_max_diff', 0) for m in MODALS if m != BASE_MODAL], default=0)
        print(f"  对齐质量: 最大时间差={max_diff:.4f}s")
        
        output_file = output_dir / f'{seq_id}.csv'
        result_df.to_csv(output_file, sep="\t", index=False)
        print(f"  已保存: {output_file}\n")


if __name__ == '__main__':
    args = parse_args()
    main(args)

