import os
from collections import deque
import numpy as np
import yaml
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.scenario.parse_object_state import parse_object_state
from omegaconf import OmegaConf
from unitraj.utils.utils import set_seed

def get_final_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    method_file = os.path.join(os.path.dirname(config_file), "method", f"{raw['defaults'][0]['method']}.yaml")
    with open(method_file, 'r', encoding='utf-8') as f:
        method_cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(raw)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(method_cfg))
    OmegaConf.set_struct(cfg, False)  # Open the struct
    set_seed(cfg.seed)
    return cfg


# --- 控制器参数 ---
LATERAL_PID_GAINS = (0.6, 0.0, 1.2)  # Kp, Ki, Kd
LONGITUDINAL_PID_GAINS = (2, 0, 0)  # Kp, Ki, Kd
FIXED_DT = 0.1  # (s) 仿真步长

class InferencePolicy(EnvInputPolicy):
    """
    一个专业的混合策略，通过继承 ReplayEgoCarPolicy 来实现行为切换
    - 前 WARMUP_STEPS 步: 自动使用父类 ReplayEgoCarPolicy 的 act 方法
    - WARMUP_STEPS 步后: 使用横向和纵向PID控制器，精准跟随模型生成的轨迹。
    """

    def __init__(self, obj, seed):
        super(InferencePolicy, self).__init__(obj, seed)

        self.Sim = None
        self.cfg = get_final_config(config_file="/data_set/UniTraj/unitraj/configs/config.yaml")

        self.WARMUP_STEPS = 20
        self.replan_frequency = 1

        # 初始化横向和纵向PID控制器
        self.lateral_pid = PIDController(*LATERAL_PID_GAINS)
        self.longitudinal_pid = PIDController(*LONGITUDINAL_PID_GAINS)
        self.current_state = self.engine.data_manager.current_scenario
        self.sdc_id = str(self.current_state["metadata"]["sdc_id"])

        self.last_match_idx = 0

        self._pred_traj = None
        self.hist_buffer = deque(maxlen=self.replan_frequency)

    def act(self, agent_id: str):
        """
        覆盖父类的 act 方法，实现我们自己的混合逻辑。
        """
        current_step = self.engine.episode_step
        time_index = current_step -1
        sdc_track = self.current_state["tracks"][self.sdc_id]

        # --- 阶段一: 数据采集 (完全模仿 ReplayEgoCarPolicy) ---
        if time_index <= self.WARMUP_STEPS:
            if time_index == 0:
                self.engine.agents['default_agent'].plan_traj = self.current_state['tracks'][self.sdc_id]['state']["position"][:self.WARMUP_STEPS]
            state = parse_object_state(sdc_track, time_index)

            if state["valid"]:
                self.control_object.set_position(state["position"])
                self.control_object.set_velocity(state["velocity"])
                self.control_object.set_heading_theta(state["heading"])
            return None

        # --- 阶段二: 闭环PID控制 ---
        else:
            if time_index % self.replan_frequency == 0:
                if time_index >= self.WARMUP_STEPS + self.replan_frequency+1 :
                    # self.get_hist_buffer([self.control_object.position, self.control_object.heading_theta, self.control_object.speed])

                    # 第一轮推理仿真后存储历史轨迹与原始未来轨迹进行拼接，用于推理过程
                    self.update_state(time_index+1)
                    self.current_state = self.engine.data_manager.current_scenario
                self._pred_traj = self.Sim.run_inference(self.current_state, time_index)
                # print('上三秒轨迹', list(self.hist_buffer)[-3:] if len(self.hist_buffer)>0 else sdc_track['state']["position"][18:21])
                # print('下三秒预测轨迹',self._pred_traj[0, :3, :2])
                self.last_match_idx = 0
                # self.engine.agents['default_agent'].plan_traj = np.vstack([self.engine.agents['default_agent'].plan_traj, self._pred_traj[0, :20, :3]])
                self.engine.agents['default_agent'].plan_traj = self._pred_traj[0, :, :2]

            pred_traj = self._pred_traj[0, :-1, :]

            ego_vehicle = self.control_object
            current_pos = ego_vehicle.position
            current_heading = ego_vehicle.heading_theta
            current_speed = ego_vehicle.speed
            # print('cur_pos', current_pos)
            self.get_hist_buffer([current_pos, current_heading, current_speed])

            # 找到轨迹上离自车当前位置最近的点
            distances = np.linalg.norm(pred_traj[:, :2] - current_pos, axis=1)
            match_idx = self.get_match_idx(pred_traj, distances, current_pos, current_heading)

            # 计算横向误差，并用叉乘判断方向（左/右）
            p_match = pred_traj[match_idx, :2]
            self.engine.agents['default_agent'].key_points = [p_match]

            # print("dist:", np.linalg.norm(ego_vehicle.position - pred_traj[0, :2]))
            # print('p_match',p_match)
            p_next = pred_traj[min(match_idx + 1, len(pred_traj) - 1), :2]


            vec_to_ego = current_pos - p_match
            vec_traj_dir = p_next - p_match
            # 叉乘的z分量决定了方向
            cross_product_z = vec_traj_dir[0] * vec_to_ego[1] - vec_traj_dir[1] * vec_to_ego[0]
            # lateral_error > 0 表示在轨迹右侧, < 0 表示在左侧
            lateral_error_m = np.sign(cross_product_z) * distances[match_idx]
            lateral_error = lateral_error_m / max(ego_vehicle.speed, 0.1)  # 1.0 防除 0

            # 目标速度：从轨迹上采样，根据两点间距离估算
            target_speed_idx = min(match_idx + 5, len(pred_traj) - 1)  # 向前看5个点
            # print('target_speed_idx',target_speed_idx)
            assert not np.isnan(pred_traj).any()
            p_target_speed = pred_traj[target_speed_idx, :2]
            # 估算目标点的速度
            target_speed = np.linalg.norm(p_target_speed - p_match) / ((target_speed_idx - match_idx) * FIXED_DT)
            speed_error = current_speed - target_speed

            # --- 3. 使用PID控制器计算最终动作 ---
            # 横向控制产生转向
            steering = self.lateral_pid.get_result(lateral_error)
            # 纵向控制产生油门/刹车
            throttle_brake = self.longitudinal_pid.get_result(speed_error)
            # print(steering,throttle_brake)

            # 裁剪输出到 MetaDrive 的有效范围 [-1, 1]
            steering = np.clip(steering, -1.0, 1.0)
            throttle_brake = np.clip(throttle_brake, -1.0, 1.0)
            return [steering, throttle_brake]

    def update_state(self, time_index):
        start_idx = time_index - self.replan_frequency   # 向前 10 步
        end_idx = time_index  # 当前步（不含）
        engine_traj = self.engine.data_manager.current_scenario["tracks"][self.sdc_id]['state']['position']
        engine_head = self.engine.data_manager.current_scenario["tracks"][self.sdc_id]['state']['heading']
        engine_vel = self.engine.data_manager.current_scenario["tracks"][self.sdc_id]['state']['velocity']
        length = self.engine.data_manager.current_scenario['length']
        for buf_idx, t in enumerate(range(start_idx, end_idx)):
            if 0 <= t < length:
                # **原地赋值** → 引擎内存被修改
                engine_traj[t, :2] = self.hist_buffer[buf_idx]['position']
                engine_head[t] = self.hist_buffer[buf_idx]['heading']
                engine_vel[t] = self.hist_buffer[buf_idx]['velocity']

    def get_hist_buffer(self, state):
        # 缓存历史
        [current_pos, current_heading, current_speed] = state
        state = {
            "position": current_pos,
            "heading": current_heading,
            "velocity": current_speed,
            "valid": True
        }
        self.hist_buffer.append(state)

    def get_match_idx(self, pred_traj, distances, current_pos, current_heading):
        # 当前车头方向向量
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)])
        vec_to_points = pred_traj[:, :2] - current_pos

        # 只保留车辆前方点（夹角 < 90°）
        dot = np.dot(vec_to_points, heading_vec)
        forward_mask = dot > 0

        # 再在前方点中找最近的点
        if forward_mask.any():
            match_idx = np.argmin(distances * forward_mask + (~forward_mask) * 1e6)
        else:
            match_idx = np.argmin(distances)  # 极端情况备用

        # 平滑更新
        if match_idx < self.last_match_idx:
            match_idx = self.last_match_idx
        self.last_match_idx = match_idx
        return match_idx


