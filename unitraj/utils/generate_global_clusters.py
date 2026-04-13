import os
import glob
import pickle
import numpy as np
from sklearn.cluster import KMeans
import re

def generate_global_clusters(data_dir, n_clusters=64, output_path='cluster_64_center_dict.pkl'):
    """
    生成全局聚类字典并保存为.pkl文件

    Args:
        data_dir (str): 原始场景数据目录（包含多个.pkl文件）
        n_clusters (int): 每类智能体的聚类中心数量
        output_path (str): 输出聚类字典的保存路径
    """
    # 初始化存储容器
    goals_dict = {
        'TYPE_VEHICLE': [],
        'TYPE_PEDESTRIAN': [],
        'TYPE_CYCLIST': [],
    }

    # 步骤1: 加载所有场景数据并合并目标点
    raw_dirs = data_dir.split(',')
    scene_files = []
    for d in raw_dirs:
        scene_files.extend(glob.glob(os.path.join(d, '**', '*.pkl'), recursive=True))
    filtered_files = [f for f in scene_files if re.match(r'.*/sd_nuplan_v1\.1_[0-9a-f]{16}\.pkl$', f)]
    print(f"找到 {len(filtered_files)} 个符合规则的 pkl 文件")
    for file_path in filtered_files:
        with open(file_path, 'rb') as f:
            scene_data = pickle.load(f)
        if scene_data['length'] < 91:
            continue
        for obj_id, obj_data in scene_data['tracks'].items():
            obj_type = obj_data['metadata']['type']
            if obj_type not in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
                continue
            valid_mask = np.array(obj_data['state']['valid'])  # 获取 valid 序列

            if valid_mask[:92].any():  # 检查前 92 帧（索引 0~91）是否有 True
                last_valid_idx = np.where(valid_mask[:92])[0][-1]  # 找到最后一个 True 的索引
            else:
                last_valid_idx = 91  # 默认取第 91 帧
                continue
            final_position = obj_data['state']['position'][last_valid_idx][:2]  # 选取最终坐标

            if obj_type == 'VEHICLE':
                goals_dict['TYPE_VEHICLE'].append(final_position)
            elif obj_type == 'PEDESTRIAN':
                goals_dict['TYPE_PEDESTRIAN'].append(final_position)
            elif obj_type == 'CYCLIST':
                goals_dict['TYPE_CYCLIST'].append(final_position)

    # 步骤2: 数据清洗与转换
    cluster_dict = {}
    for obj_type in goals_dict:
        points = np.array(goals_dict[obj_type])
        if len(points) == 0:
            print(f"警告: {obj_type}数据为空，跳过聚类")
            continue
        # 去除无效点（假设无效点为[0,0]）
        valid_mask = (points != [0, 0]).any(axis=1)
        filtered_points = points[valid_mask]

        print(f"聚类类型: {obj_type}, 有效点数: {len(filtered_points)}")

        # 步骤3: 执行聚类
        if len(filtered_points) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
            kmeans.fit(filtered_points)
            cluster_centers = kmeans.cluster_centers_
        elif len(filtered_points) > 0:
            print(f"警告: {obj_type}数据不足，使用随机采样补全")
            cluster_centers = filtered_points[
                np.random.choice(len(filtered_points), min(len(filtered_points), n_clusters), replace=True)]
        else:
            cluster_centers = np.zeros((n_clusters, 2))  # 用全 0 填充

        cluster_dict[obj_type] = np.array(cluster_centers)  # 确保存入的是 np.ndarray

    # 步骤4: 保存聚类字典
    with open(output_path, 'wb') as f:
        pickle.dump(cluster_dict, f)
    print(f"聚类字典已保存至 {output_path}")


# 使用示例
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成全局聚类字典")
    parser.add_argument('--data_dir', type=str, required=True, help='原始场景数据目录（包含多个.pkl文件）')
    parser.add_argument('--n_clusters', type=int, default=64, help='每类智能体的聚类中心数量')
    parser.add_argument('--output_path', type=str, required=True, help='输出聚类字典的保存路径')
    args = parser.parse_args()

    generate_global_clusters(
        data_dir=args.data_dir,
        n_clusters=args.n_clusters,
        output_path=args.output_path
    )