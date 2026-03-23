import numpy as np

def find_gt(gt_data, timestamp_lidar):
    time_diff_min = np.inf
    for timestamp, data in gt_data.items():
        time_diff = np.abs(float(timestamp_lidar) - float(timestamp))
        if time_diff < time_diff_min:
            gt_data = data
            time_diff_min = time_diff
    return gt_data

def extract_feature_set(data, labels, time_ind, gt, skip_empty_frames=True, 
                        window_velocity=None, window_acceleration=None):
    """
    提取特征集
    
    Args:
        data: 点云数据
        labels: 聚类标签
        time_ind: 时间索引
        gt: ground truth位置
        skip_empty_frames: 是否跳过全零帧的聚类，默认True（跳过所有帧都是空的聚类）
        window_velocity: 窗口整体的速度（如果当前窗口无法计算，使用此值）
        window_acceleration: 窗口整体的加速度（如果当前窗口无法计算，使用此值）
    
    Returns:
        feature_set_all: 特征集数组
        label_set_all: 标签集数组（基于三轴误差都<1.5米的判断）
        weighted_centers: 每个聚类的有效帧加权平均位置（用于判断正样本）
    """
    unique_labels = set(labels)
    unique_time_ind = sorted(set(time_ind))
    # 确保包含所有应该存在的时间帧
    # 窗口固定为4帧，时间索引为1,2,3,4
    if len(time_ind) > 0:
        # 假设窗口固定为4帧（根据build_livox_avia_dataset.py的累积4帧逻辑）
        expected_time_ind = [1, 2, 3, 4]  # 固定4帧窗口
        # 合并实际存在和应该存在的时间索引
        all_time_ind = sorted(set(unique_time_ind + expected_time_ind))
    else:
        all_time_ind = [1, 2, 3, 4]  # 如果没有数据，也假设4帧
    feature_set_list = []
    label_cluster_list = []
    weighted_center_list = []  # 存储每个聚类的有效帧加权平均位置
    valid_label_list = []  # 存储未被跳过的聚类标签，用于后续判断
    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            continue
        feature_set = np.array([])
        class_member_mask = labels == k
        
        # 首先提取所有时间帧的位置均值，用于零帧填充
        time_positions = []
        valid_position_mask = []
        for idx, time in enumerate(all_time_ind):
            time_mask = time_ind == time
            masked_data = data[class_member_mask & time_mask]
            if np.size(masked_data) != 0:
                xyz_mean = np.mean(masked_data, axis=0)
                time_positions.append(xyz_mean)
                valid_position_mask.append(True)
            else:
                time_positions.append(np.zeros(3))
                valid_position_mask.append(False)
        
        # 检查是否有空帧
        time_positions = np.array(time_positions)
        valid_position_mask = np.array(valid_position_mask)
        time_indices = np.arange(len(all_time_ind))  # 时间索引，用于时间差加权
        
        # 在零帧填充前计算有效帧的加权平均位置（用于判断正样本）
        # 权重为每帧的数据点数量
        valid_positions_for_center = time_positions[valid_position_mask]
        valid_frame_point_counts = []
        for idx, is_valid in enumerate(valid_position_mask):
            if is_valid:
                time = all_time_ind[idx]
                time_mask = time_ind == time
                frame_point_count = np.sum(class_member_mask & time_mask)
                valid_frame_point_counts.append(frame_point_count)
        
        # 计算加权平均位置
        # 有效帧只有一帧就用这一帧位置，多帧则用加权平均
        if len(valid_positions_for_center) == 0:
            # 如果所有帧都是空的，使用零向量
            weighted_center = np.zeros(3)
        elif len(valid_positions_for_center) == 1:
            # 有效帧只有一帧，直接用这一帧位置
            weighted_center = valid_positions_for_center[0]
        else:
            # 多帧有效，使用加权平均（权重为每帧的数据点数量）
            valid_frame_point_counts = np.array(valid_frame_point_counts)
            total_points = np.sum(valid_frame_point_counts)
            if total_points > 0:
                # 加权平均：权重为数据点数量
                weighted_center = np.average(valid_positions_for_center, axis=0, weights=valid_frame_point_counts)
            else:
                # 如果没有数据点，使用简单平均
                weighted_center = np.mean(valid_positions_for_center, axis=0)
        
        # 只有全零帧的聚类才跳过（所有帧都是空的）
        if skip_empty_frames and np.all(~valid_position_mask):
            # 所有帧都是空的，跳过这个聚类（不添加到列表）
            continue
        
        # 添加到加权中心列表（只有在不跳过的情况下）
        weighted_center_list.append(weighted_center)
        valid_label_list.append(k)  # 记录未被跳过的聚类标签
        
        # 在零帧填充前计算速度和加速度（基于有效帧）
        # 计算窗口整体的速度和加速度
        valid_indices_for_velocity = time_indices[valid_position_mask]
        cluster_velocity = np.zeros(3)
        cluster_acceleration = np.zeros(3)
        
        if len(valid_indices_for_velocity) >= 2:
            # 计算速度：基于有效帧的位置序列
            valid_positions = time_positions[valid_indices_for_velocity]
            # 计算相邻有效帧之间的速度
            velocities = []
            for i in range(len(valid_indices_for_velocity) - 1):
                idx1 = valid_indices_for_velocity[i]
                idx2 = valid_indices_for_velocity[i + 1]
                pos1 = valid_positions[i]
                pos2 = valid_positions[i + 1]
                # 时间差（假设每帧间隔为1）
                dt = idx2 - idx1
                if dt > 0:
                    velocity = (pos2 - pos1) / dt
                    velocities.append(velocity)
            
            if len(velocities) > 0:
                # 窗口整体的速度：所有速度的平均值
                cluster_velocity = np.mean(velocities, axis=0)
                
                # 计算加速度：基于速度序列
                if len(velocities) >= 2:
                    accelerations = []
                    for i in range(len(velocities) - 1):
                        v1 = velocities[i]
                        v2 = velocities[i + 1]
                        idx1 = valid_indices_for_velocity[i]
                        idx2 = valid_indices_for_velocity[i + 1]
                        dt = idx2 - idx1
                        if dt > 0:
                            acceleration = (v2 - v1) / dt
                            accelerations.append(acceleration)
                    
                    if len(accelerations) > 0:
                        # 窗口整体的加速度：所有加速度的平均值
                        cluster_acceleration = np.mean(accelerations, axis=0)
        
        # 如果无法计算速度，使用传入的窗口速度（来自最近的窗口）
        if len(valid_indices_for_velocity) < 2 or np.all(np.abs(cluster_velocity) < 1e-6):
            if window_velocity is not None and np.any(np.abs(window_velocity) > 1e-6):
                cluster_velocity = window_velocity
        
        # 如果无法计算加速度，使用传入的窗口加速度（来自最近的窗口）
        if len(valid_indices_for_velocity) < 3 or np.all(np.abs(cluster_acceleration) < 1e-6):
            if window_acceleration is not None and np.any(np.abs(window_acceleration) > 1e-6):
                cluster_acceleration = window_acceleration
        
        # 提取每个时间帧的完整特征（15维）
        # 零帧用0值填充，不跳过
        for idx, time in enumerate(all_time_ind):
            time_mask = time_ind == time
            masked_data = data[class_member_mask & time_mask]
            
            # 计算位置均值、标准差和范围
            if np.size(masked_data) != 0:
                xyz_mean = np.mean(masked_data, axis=0)
                xyz_std = np.std(masked_data, axis=0)
                xyz_range = np.max(masked_data, axis=0) - np.min(masked_data, axis=0)
            else:
                # 如果是零帧，用0值填充（和旧版逻辑一样）
                xyz_mean = np.zeros(3)
                xyz_std = np.zeros(3)
                xyz_range = np.zeros(3)
            
            # 组合15维特征: mean(3) + std(3) + range(3) + velocity(3) + acceleration(3)
            # 速度和加速度是窗口整体的特征，每个时间帧都使用相同的值
            feature = np.concatenate((
                xyz_mean,           # 3维：位置均值
                xyz_std,            # 3维：位置标准差
                xyz_range,          # 3维：位置范围
                cluster_velocity,   # 3维：窗口整体速度
                cluster_acceleration # 3维：窗口整体加速度
            ), axis=0).reshape(1,-1)
            
            if np.size(feature_set) == 0:
                feature_set = feature
            else:
                feature_set = np.vstack([feature_set, feature])
        feature_set_list.append(feature_set)
        
    # 使用有效帧加权平均位置判断正样本
    # 判断标准：x,y,z三轴误差都<1.5米
    # 注意：valid_label_list只包含未被跳过的聚类标签（与feature_set_list长度一致）
    for idx in range(len(weighted_center_list)):
        weighted_center = weighted_center_list[idx]
        
        # 计算三轴误差
        x_error = np.abs(weighted_center[0] - gt[0])
        y_error = np.abs(weighted_center[1] - gt[1])
        z_error = np.abs(weighted_center[2] - gt[2])
        
        # 三轴误差都小于1.5米则为正样本
        if x_error < 1.5 and y_error < 1.5 and z_error < 1.5:
            label_cluster = 1
        else:
            label_cluster = 0
                
        label_cluster_list.append(label_cluster)

    # 检查是否有有效的聚类
    if len(feature_set_list) == 0:
        # 如果没有有效聚类，返回空数组
        return np.array([]), np.array([]), np.array([])
    
    # 找到最大的时间维度（不同聚类可能有不同数量的有效帧）
    max_time_dim = max(fs.shape[0] if len(fs.shape) > 0 and fs.shape[0] > 0 else 0 for fs in feature_set_list)
    
    # 将所有特征集填充到相同的最大时间维度
    padded_feature_set_list = []
    for fs in feature_set_list:
        if len(fs.shape) > 0 and fs.shape[0] > 0:
            if fs.shape[0] < max_time_dim:
                # 用零填充到最大时间维度
                padding_shape = (max_time_dim - fs.shape[0], fs.shape[1])
                padding = np.zeros(padding_shape, dtype=fs.dtype)
                fs_padded = np.vstack([fs, padding])
            else:
                fs_padded = fs
        else:
            # 空特征集，创建全零的特征集
            fs_padded = np.zeros((max_time_dim, 15), dtype=np.float32)
        padded_feature_set_list.append(fs_padded)
    
    feature_set_all = np.stack(padded_feature_set_list, axis = 0)
    label_set_all = np.array(label_cluster_list)
    weighted_centers = np.array(weighted_center_list)
    return feature_set_all, label_set_all, weighted_centers



def extract_feature_set_predict(data, labels, time_ind, skip_empty_frames=True,
                                window_velocity=None, window_acceleration=None):
    """
    提取特征集（用于预测）
    
    Args:
        data: 点云数据
        labels: 聚类标签
        time_ind: 时间索引
        skip_empty_frames: 是否跳过全零帧的聚类，默认True（跳过所有帧都是空的聚类）
        window_velocity: 窗口整体的速度（如果当前窗口无法计算，使用此值）
        window_acceleration: 窗口整体的加速度（如果当前窗口无法计算，使用此值）
    """
    unique_labels = set(labels)
    unique_time_ind = sorted(set(time_ind))
    # 确保包含所有应该存在的时间帧
    # 窗口固定为4帧，时间索引为1,2,3,4
    if len(time_ind) > 0:
        # 假设窗口固定为4帧（根据build_livox_avia_dataset.py的累积4帧逻辑）
        expected_time_ind = [1, 2, 3, 4]  # 固定4帧窗口
        # 合并实际存在和应该存在的时间索引
        all_time_ind = sorted(set(unique_time_ind + expected_time_ind))
    else:
        all_time_ind = [1, 2, 3, 4]  # 如果没有数据，也假设4帧
    feature_set_list = []
    cluster_label_list = []
    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            continue
        feature_set = np.array([])
        class_member_mask = labels == k
        
        # 首先提取所有时间帧的位置均值，用于零帧填充
        time_positions = []
        valid_position_mask = []
        for idx, time in enumerate(all_time_ind):
            time_mask = time_ind == time
            masked_data = data[class_member_mask & time_mask]
            if np.size(masked_data) != 0:
                xyz_mean = np.mean(masked_data, axis=0)
                time_positions.append(xyz_mean)
                valid_position_mask.append(True)
            else:
                time_positions.append(np.zeros(3))
                valid_position_mask.append(False)
        
        # 检查是否有空帧
        time_positions = np.array(time_positions)
        valid_position_mask = np.array(valid_position_mask)
        time_indices = np.arange(len(all_time_ind))  # 时间索引，用于时间差加权
        
        # 在零帧填充前计算速度和加速度（基于有效帧）
        # 计算窗口整体的速度和加速度
        valid_indices_for_velocity = time_indices[valid_position_mask]
        cluster_velocity = np.zeros(3)
        cluster_acceleration = np.zeros(3)
        
        if len(valid_indices_for_velocity) >= 2:
            # 计算速度：基于有效帧的位置序列
            valid_positions = time_positions[valid_indices_for_velocity]
            # 计算相邻有效帧之间的速度
            velocities = []
            for i in range(len(valid_indices_for_velocity) - 1):
                idx1 = valid_indices_for_velocity[i]
                idx2 = valid_indices_for_velocity[i + 1]
                pos1 = valid_positions[i]
                pos2 = valid_positions[i + 1]
                # 时间差（假设每帧间隔为1）
                dt = idx2 - idx1
                if dt > 0:
                    velocity = (pos2 - pos1) / dt
                    velocities.append(velocity)
            
            if len(velocities) > 0:
                # 窗口整体的速度：所有速度的平均值
                cluster_velocity = np.mean(velocities, axis=0)
                
                # 计算加速度：基于速度序列
                if len(velocities) >= 2:
                    accelerations = []
                    for i in range(len(velocities) - 1):
                        v1 = velocities[i]
                        v2 = velocities[i + 1]
                        idx1 = valid_indices_for_velocity[i]
                        idx2 = valid_indices_for_velocity[i + 1]
                        dt = idx2 - idx1
                        if dt > 0:
                            acceleration = (v2 - v1) / dt
                            accelerations.append(acceleration)
                    
                    if len(accelerations) > 0:
                        # 窗口整体的加速度：所有加速度的平均值
                        cluster_acceleration = np.mean(accelerations, axis=0)
        
        # 如果无法计算速度，使用传入的窗口速度（来自最近的窗口）
        if len(valid_indices_for_velocity) < 2 or np.all(np.abs(cluster_velocity) < 1e-6):
            if window_velocity is not None and np.any(np.abs(window_velocity) > 1e-6):
                cluster_velocity = window_velocity
        
        # 如果无法计算加速度，使用传入的窗口加速度（来自最近的窗口）
        if len(valid_indices_for_velocity) < 3 or np.all(np.abs(cluster_acceleration) < 1e-6):
            if window_acceleration is not None and np.any(np.abs(window_acceleration) > 1e-6):
                cluster_acceleration = window_acceleration
        
        # 只有全零帧的聚类才跳过（所有帧都是空的）
        if skip_empty_frames and np.all(~valid_position_mask):
            # 所有帧都是空的，跳过这个聚类
            continue
        
        # 提取每个时间帧的完整特征（15维）
        # 零帧用0值填充，不跳过
        for idx, time in enumerate(all_time_ind):
            time_mask = time_ind == time
            masked_data = data[class_member_mask & time_mask]
            
            # 计算位置均值、标准差和范围
            if np.size(masked_data) != 0:
                xyz_mean = np.mean(masked_data, axis=0)
                xyz_std = np.std(masked_data, axis=0)
                xyz_range = np.max(masked_data, axis=0) - np.min(masked_data, axis=0)
            else:
                # 如果是零帧，用0值填充（和旧版逻辑一样）
                xyz_mean = np.zeros(3)
                xyz_std = np.zeros(3)
                xyz_range = np.zeros(3)
            
            # 组合15维特征: mean(3) + std(3) + range(3) + velocity(3) + acceleration(3)
            # 速度和加速度是窗口整体的特征，每个时间帧都使用相同的值
            feature = np.concatenate((
                xyz_mean,           # 3维：位置均值
                xyz_std,            # 3维：位置标准差
                xyz_range,          # 3维：位置范围
                cluster_velocity,   # 3维：窗口整体速度
                cluster_acceleration # 3维：窗口整体加速度
            ), axis=0).reshape(1,-1)
            
            if np.size(feature_set) == 0:
                feature_set = feature
            else:
                feature_set = np.vstack([feature_set, feature])
        feature_set_list.append(feature_set)
        cluster_label_list.append(k)

    # 检查是否有有效的聚类
    if len(feature_set_list) == 0:
        # 如果没有有效聚类，返回空数组
        return np.array([]), np.array([])

    # 找到最大的时间维度（不同聚类可能有不同数量的有效帧）
    max_time_dim = max(fs.shape[0] if len(fs.shape) > 0 and fs.shape[0] > 0 else 0 for fs in feature_set_list)
    
    # 将所有特征集填充到相同的最大时间维度
    padded_feature_set_list = []
    for fs in feature_set_list:
        if len(fs.shape) > 0 and fs.shape[0] > 0:
            if fs.shape[0] < max_time_dim:
                # 用零填充到最大时间维度
                padding_shape = (max_time_dim - fs.shape[0], fs.shape[1])
                padding = np.zeros(padding_shape, dtype=fs.dtype)
                fs_padded = np.vstack([fs, padding])
            else:
                fs_padded = fs
        else:
            # 空特征集，创建全零的特征集
            fs_padded = np.zeros((max_time_dim, 15), dtype=np.float32)
        padded_feature_set_list.append(fs_padded)

    feature_set_all = np.stack(padded_feature_set_list, axis = 0)
    cluster_label_set_all = np.array(cluster_label_list).reshape(-1,1)
    return feature_set_all, cluster_label_set_all
