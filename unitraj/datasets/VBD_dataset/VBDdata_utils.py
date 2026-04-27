import torch
import numpy as np
from metadrive.type import MetaDriveType

def wrap_to_pi(angle):
    """
    Wrap an angle to the range [-pi, pi].

    Args:
        angle (float): The input angle.

    Returns:
        float: The wrapped angle.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
    
def calculate_relations(agents, polylines, traffic_lights):
    """
    Calculate the relations between agents, polylines, and traffic lights.

    Args:
        agents (numpy.ndarray): Array of agent positions and orientations.
        polylines (numpy.ndarray): Array of polyline positions.
        traffic_lights (numpy.ndarray): Array of traffic light positions.

    Returns:
        numpy.ndarray: Array of relations between the elements.
    """
    n_agents = agents.shape[0]
    n_polylines = polylines.shape[0]
    n_traffic_lights = traffic_lights.shape[0]
    n = n_agents + n_polylines + n_traffic_lights

    # Prepare a single array to hold all elements
    all_elements = np.concatenate([
        agents[:, -1, :3], # 提取车辆的最后一个时间步的位置和方向 x, y, theta
        polylines[:, 0, :3], # 提取道路的第一个点的位置和方向 x, y, theta
        np.concatenate([traffic_lights[:, :2], np.zeros((n_traffic_lights, 1))], axis=1) # 提取交通灯的位置 x, y，并在第三维补充零
    ], axis=0)

    # Compute pairwise differences using broadcasting
    pos_diff = all_elements[:, :2][:, None, :] - all_elements[:, :2][None, :, :] # 计算位置差

    # Compute local positions and angle differences
    cos_theta = np.cos(all_elements[:, 2])[:, None]
    sin_theta = np.sin(all_elements[:, 2])[:, None]
    local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
    local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta
    theta_diff = wrap_to_pi(all_elements[:, 2][:, None] - all_elements[:, 2][None, :])

    # Set theta_diff to zero for traffic lights
    start_idx = n_agents + n_polylines
    theta_diff = np.where((np.arange(n) >= start_idx)[:, None] | (np.arange(n) >= start_idx)[None, :], 0, theta_diff)

    # Set the diagonal of the differences to a very small value
    diag_mask = np.eye(n, dtype=bool)
    epsilon = 0.01
    local_pos_x = np.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = np.where(diag_mask, epsilon, local_pos_y)
    theta_diff = np.where(diag_mask, epsilon, theta_diff)

    # Conditions for zero coordinates
    zero_mask = np.logical_or(all_elements[:, 0][:, None] == 0, all_elements[:, 0][None, :] == 0)

    # Initialize relations array
    relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)

    # Apply zero mask
    relations = np.where(zero_mask[..., None], 0.0, relations)

    return relations

def data_process_agent(
    scenario,
    max_num_objects=64,
    current_index=10,
    use_log=True,
    selected_agents=None,
    remove_history=False,
):
    """
    Process the data for surrounding agents in a given scenario.

    Args:
        scenario (datatypes.SimulatorState): The simulator state containing the agent data.
        max_num_objects (int): The maximum number of objects to consider.
        current_index (int): The current time index.
        use_log (bool): Whether to use log trajectory or sim trajectory.
        selected_agents (list[int] or None): List of agent IDs to consider. If None, all agents will be considered.

    Returns:
        tuple: A tuple containing the processed agent data, including:
            - agents_history (ndarray): The history of agent trajectories. Shape: (max_object, history_length, 8)
            - agents_future (ndarray): The future agent trajectories. Shape: (max_object, future_length, 5)
            - agents_interested (ndarray): The interest level of agents. Shape: (max_object,)
            - agents_type (ndarray): The type of agents. Shape: (max_object,)
    """
    if use_log:
        log_trajectory = scenario['tracks']
    else:
        log_trajectory = scenario['sim_trajectory']

    metadata = scenario.get('metadata', {})
    sdc_id = metadata['sdc_id']

    """根据移动距离筛选代理"""
    object_summary = metadata.get('object_summary', {})
    # moving_distances = {
    #     track_id: object_summary.get(track_id, {}).get('moving_distance', 0)
    #     for track_id in log_trajectory
    # }
    # current_threshold = 3
    # # 动态调整阈值直到找到足够agent或阈值降为0
    # while current_threshold >= 0:
    #     valid_agents = {
    #         agent_id: data
    #         for agent_id, data in log_trajectory.items()
    #         if moving_distances.get(agent_id, 0) >= current_threshold  # 应用当前阈值
    #     }
    #     # 满足数量要求则退出循环
    #     if len(valid_agents) >= 16:
    #         break
    #     # 阈值递减但保持非负
    #     current_threshold = max(current_threshold - 1, 0)

    """根据agent的type属性筛选动态代理"""
    excluded_types = {'TRAFFIC_BARRIER', 'TRAFFIC_CONE'}
    valid_agents0 = {
        agent_id: data
        for agent_id, data in log_trajectory.items()
        if data['type'] not in excluded_types  # Exclude agents with specified types
    }

    """根据valid_length长度筛选有效移动代理"""
    valid_length = {
        track_id: object_summary.get(track_id, {}).get('valid_length', 0)
        for track_id in log_trajectory
    }

    valid_agents = {
        agent_id: data
        for agent_id, data in valid_agents0.items()
        if valid_length.get(agent_id, 0) >= 20  # 过滤掉 valid_length < 20 的 agent
    }
    # 计算所有有效代理的位置
    positions = []
    agent_ids_filtered = []
    for agent_id, data in valid_agents.items():
        try:
            position = np.array(data['state']['position'][current_index])[:2]  # 取前两维
            # 如果 position 为 (0,0)，则寻找第一个非 (0,0) 的位置
            if np.all(position == 0):
                for pos in data['state']['position']:
                    pos_2d = np.array(pos)[:2]  # 取前两维
                    if not np.all(pos_2d == 0):  # 发现非 (0,0) 值
                        position = pos_2d
                        break
            positions.append(position)
            agent_ids_filtered.append(agent_id)  # 记录有效代理 ID
        except (KeyError, IndexError) as e:
            print('Error:', e)
            print('agent 处理有问题')
            continue
    # 将所有位置合并成一个NumPy数组
    positions_array = np.array(positions)

    # calculate distance to sdc
    if selected_agents is None:
        sdc_position = np.asarray(log_trajectory[sdc_id]['state']['position'][current_index][:2])
        agents_positions = positions_array #（num_agents, 2）
        distance_to_sdc = np.linalg.norm(agents_positions - sdc_position, axis=-1)
        sorted_indices = np.argsort(distance_to_sdc)[:max_num_objects] # 距主车最近的max_num_objects个代理的索引
        agent_ids = [agent_ids_filtered[i] for i in sorted_indices]
    else:
        agent_ids = selected_agents
            
    ############# Get agents' trajectory #############
    # feature: x, y, yaw, velx, vely, length, width, height
    agents_history = np.zeros((max_num_objects, current_index+1, 8), dtype=np.float32)
    agents_type = np.zeros((max_num_objects,), dtype=np.int32)
    agents_interested = np.zeros((max_num_objects,), dtype=np.int32)
    # agents_future = np.zeros((max_num_objects, log_trajectory[sdc_id]['state']['position'].shape[0]-current_index, 5), dtype=np.float32)
    agents_future = np.zeros((max_num_objects, 91-current_index, 5), dtype=np.float32)
    agent_type_mapping = {
        'VEHICLE': 1,
        'PEDESTRIAN': 2,
        'CYCLIST': 3,
        # 'TRAFFIC_BARRIER': 4,
        # 'TRAFFIC_CONE': 4,
    }
    for i, a in enumerate(agent_ids):
        log_trajectory_a = valid_agents[a]
        agent_type = log_trajectory_a['type']
        valid = log_trajectory_a['state']['valid'][current_index]

        if not valid:
            agents_interested[i] = 0
            continue
            
        # if metadata.is_modeled[a] or metadata.objects_of_interest[a]: # 咋改
        #     agents_interested[i] = 10
        # else:
        #     agents_interested[i] = 1
        agents_interested[i] = 1
        agents_type[i] = agent_type_mapping.get(agent_type, 0)

        keys = ['position', 'heading', 'velocity', 'length', 'width', 'height']
        for k in keys:
            if isinstance(log_trajectory_a['state'][k], list):
                log_trajectory_a['state'][k] = np.array(log_trajectory_a['state'][k])

        agents_history[i] = np.column_stack([
               log_trajectory_a['state']['position'][:current_index+1, 0], # x
               log_trajectory_a['state']['position'][:current_index+1, 1], # y
               log_trajectory_a['state']['heading'][:current_index+1],
               log_trajectory_a['state']['velocity'][:current_index+1, 0], # velx
               log_trajectory_a['state']['velocity'][:current_index+1, 1], # vely
               log_trajectory_a['state']['length'][:current_index+1],
               log_trajectory_a['state']['width'][:current_index+1],
               log_trajectory_a['state']['height'][:current_index+1],
            ])

        agents_history[i][log_trajectory_a['state']['valid'][:current_index+1] == False] = 0

        agents_future[i] = np.column_stack([
               log_trajectory_a['state']['position'][current_index:91, 0],
               log_trajectory_a['state']['position'][current_index:91, 1],
               log_trajectory_a['state']['heading'][current_index:91],
               log_trajectory_a['state']['velocity'][current_index:91, 0],
               log_trajectory_a['state']['velocity'][current_index:91, 1]
            ])

        agents_future[i][log_trajectory_a['state']['valid'][current_index:91] == False] = 0

    # Remove history
    if remove_history:
        agents_history[:, :-1] = 0
    
    
    return agents_history, agents_future, agents_interested, agents_type, sorted_indices, sdc_position

def data_process_traffic_light(
    scenario,
    current_index = 10,
):
    """
    Process traffic light data from the given scenario.

    Args:
        scenario (datatypes.SimulatorState): The simulator state containing traffic light information.

    Returns:
        tuple: A tuple containing the processed traffic light points, lane IDs, and states.
    """
    traffic_lights = scenario['dynamic_map_states']
    if not traffic_lights :
        return np.zeros((16, 3), dtype=np.float32), np.zeros((16,), dtype=np.int32), np.zeros((16,), dtype=np.int32)
    ############# Get Traffic Lights #############
    object_states = []
    stop_points = []
    lane_ids = []
    for agent_id, data in traffic_lights.items():
        try:
            object_states.append(data['state']['object_state'][current_index])
            stop_points.append(data['stop_point'][:2])
            lane_ids.append(data['lane'] if data.get('lane') else agent_id)
        except (KeyError, IndexError) as e:
            print('Error:', e)
            print('traffic_lights处理有问题')
            # 错误处理：如缺少关键数据
            continue
    traffic_lane_ids = np.array(lane_ids)

    state_mapping = {
        'TRAFFIC_LIGHT_GREEN': 1,
        'TRAFFIC_LIGHT_YELLOW': 2,
        'TRAFFIC_LIGHT_RED': 3,
        'TRAFFIC_LIGHT_UNKNOWN': 0
    }
    status = [MetaDriveType.simplify_light_status(es) for es in object_states]
    numeric_states = [state_mapping.get(s, 3) for s in status]
    traffic_light_states = np.array(numeric_states, dtype=np.int32) # (num_lights, 1)
    traffic_stop_points = np.asarray(stop_points) # (num_lights, 2)
    # traffic_light_valid = np.asarray(traffic_lights.valid)[:, current_index] # nuplan没有这一项
        
    # traffic_light_points = np.concatenate([traffic_stop_points, traffic_light_states[:, None]], axis=1) #替换为下面的代码
    traffic_light_points = np.zeros((16, 3), dtype=np.float32)
    traffic_light_points[:min(16, len(traffic_stop_points))] = np.concatenate(
        [traffic_stop_points, traffic_light_states[:, None]], axis=1)[:16]
    traffic_light_points = np.float32(traffic_light_points) # (num_lights, 3)
    # traffic_light_points = np.where(
    #     traffic_light_valid[:, None],
    #     traffic_light_points,
    #     0.0
    # )
        
    return traffic_light_points, traffic_lane_ids, traffic_light_states

def data_process_scenario(
    scenario,
    max_num_objects = 64,
    max_polylines = 256,
    current_index = 10,
    num_points_polyline = 30,
    use_log = True,
    selected_agents = None,
    remove_history = False,
):
    """
    Process the data for a given scenario.

    Args:
        scenario (datatypes.SimulatorState): The simulator state containing the scenario data.

    Returns:
        dict: A dictionary containing the processed data.
    """
    (agents_history, agents_future, agents_interested, agents_type, agents_id, sdc_position) = data_process_agent(
        scenario,
        max_num_objects = max_num_objects,
        current_index = current_index,
        use_log = use_log,
        selected_agents = selected_agents,
        remove_history=remove_history,
    )
                
    (traffic_light_points, traffic_lane_ids, traffic_light_states) = data_process_traffic_light(
        scenario,
        current_index = current_index,
    )

    lane_type_mapping = {
        "LANE_SURFACE_STREET": 2,
        "LANE_SURFACE_UNSTRUCTURE": 4,  # 这里用 -1 表示未定义，视需求调整
        "LANE_UNKNOWN": 0,
        "LANE_FREEWAY": 1,
        "LANE_BIKE_LANE": 3,

        "ROAD_LINE_BROKEN_SINGLE_WHITE": 6,
        "ROAD_LINE_SOLID_SINGLE_WHITE": 7,
        "ROAD_LINE_SOLID_DOUBLE_WHITE": 8,
        "ROAD_LINE_BROKEN_SINGLE_YELLOW": 9,
        "ROAD_LINE_BROKEN_DOUBLE_YELLOW": 10,
        "ROAD_LINE_SOLID_SINGLE_YELLOW": 11,
        "ROAD_LINE_SOLID_DOUBLE_YELLOW": 12,
        "ROAD_LINE_PASSING_DOUBLE_YELLOW": 13,

        "ROAD_EDGE_BOUNDARY": 15,
        "ROAD_EDGE_MEDIAN": 16,
        "ROAD_EDGE_SIDEWALK": 14,  # 可能无对应数值，可设为特殊值
        "STOP_SIGN": 17,
        "CROSSWALK": 18,
        "SPEED_BUMP": 19,
        "DRIVEWAY": 5,  # 未在数值映射中，设为特殊值
        # "GUARDRAIL": -4  # 未在数值映射中，设为特殊值
    }

    # 导入nuplan的polyline，进行采样处理得到[100,100,5]的polylines
    polylines = []
    polyline_importance_scores = []
    threshold_distance = 150  # 120m
    for id, data in scenario['map_features'].items():
        try:
            polyline = np.array(data.get('polyline', []))
            if len(polyline) < 3:
                # print('跳过点数不足的多段线')
                polygon = np.array(data.get('polygon', []))
                if len(polygon) > 0 :
                    polyline = polygon
                else:
                    continue
            # 计算与SDC的最小距离（用于重要性评估）
            min_distance = float('inf')
            for point in polyline:
                distance = np.linalg.norm(point[:2] - sdc_position)
                min_distance = min(min_distance, distance)
            # 距离过滤
            if min_distance > threshold_distance:
                continue  # 超过 150m，跳过该 polyline

            p_x= polyline[:, 0]
            p_y= polyline[:, 1]
            # add heading
            if polyline.shape[1] > 2:
                heading = polyline[:, 2]
            else:
                dx = polyline[1:, 0] - polyline[:-1, 0]
                dy = polyline[1:, 1] - polyline[:-1, 1]
                epsilon = 1e-6
                heading = np.arctan2(dy + epsilon, dx + epsilon)
                heading = np.append(heading, heading[-1]) if len(heading) > 0 else np.zeros(1)

            lane_type =lane_type_mapping.get(data['type'], 0)
            lane_type = np.repeat(lane_type, len(p_x))

            if id in traffic_lane_ids.tolist():
                idx = np.argmax(traffic_lane_ids == id)  # 找到第一个匹配的索引
                traffic_light_state = traffic_light_states[idx]
            else:
                traffic_light_state = 0
            traffic_light_state = np.repeat(traffic_light_state, len(p_x))

            polyline = np.stack([p_x, p_y, heading, traffic_light_state, lane_type], axis=1)
            # sample points and fill into fixed-size array
            polyline_len = polyline.shape[0]
            sampled_points = np.linspace(0, polyline_len - 1, num_points_polyline, dtype=np.int32)
            cur_polyline = np.take(polyline, sampled_points, axis=0)

            score = 1.0 / (min_distance + 1.0)
            polylines.append(cur_polyline)
            polyline_importance_scores.append(score)

        except (KeyError, IndexError) as e:
            print('Error:', e)
            print('polyline处理有问题')
            continue

    if len(polylines) > max_polylines:
        # 获取重要性最高的索引
        top_indices = np.argsort(polyline_importance_scores)[::-1][:max_polylines]
        polylines = [polylines[i] for i in top_indices]
        polylines_valid = np.ones((max_polylines,), dtype=np.int32)
    else:
        # 填充到最大数量
        polylines = polylines + [np.zeros((num_points_polyline, 5), dtype=np.float32)] * (
                    max_polylines - len(polylines))
        polylines_valid = np.ones((len(polylines),), dtype=np.int32)
        polylines_valid = np.pad(polylines_valid, (0, max_polylines - len(polylines_valid)))
    # 转换为numpy数组
    polylines = np.array(polylines, dtype=np.float32)

    relations = calculate_relations(agents_history, polylines, traffic_light_points)
    relations = np.asarray(relations)
    
    data_dict = {
        'agents_history': np.float32(agents_history),
        'agents_interested': np.int32(agents_interested),
        'agents_type': np.int32(agents_type),
        'agents_future': np.float32(agents_future), 
        'traffic_light_points': np.float32(traffic_light_points),
        'polylines': np.float32(polylines),
        'polylines_valid': np.int32(polylines_valid),
        'relations': np.float32(relations),
        'agents_id': np.int32(agents_id),
    }
    return data_dict

def data_collate_fn(batch_list):
    """
    Collects a batch of data from a list of transitions.

    Args:
        batch_list (List): a list of transitions.

    Returns:
        Dict[str, torch.Tensor]: a batch of data.
    """
    list_len = len(batch_list)
    key_to_list = {}
    for key in batch_list[0].keys():
        key_to_list[key] = [batch_list[i][key] for i in range(list_len)]
        
    input_batch = {}
    for key, value in key_to_list.items():
        if 'scenario' not in key:
            # print(f"Key: {key}, Shapes: {[v.shape for v in value]}")
            input_batch[key] = torch.from_numpy(np.stack(value, axis=0))
        else:
            input_batch[key] = value
            
    return input_batch