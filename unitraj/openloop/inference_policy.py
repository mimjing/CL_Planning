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

class OpenInferencePolicy(EnvInputPolicy):
    """
    纯开环：第 10 帧一次性预测整条轨迹，后续直接回放，不再做任何闭环修正。
    """

    def __init__(self, obj, seed):
        super().__init__(obj, seed)
        self.Sim = None
        self.cfg = get_final_config(config_file="/data_set/UniTraj/unitraj/configs/config.yaml")

        # 预测起点固定在第 10 帧（可改）
        self.PREDICT_FRAME = 20
        # 预测总长度（含当前帧往后）
        self.PRED_LEN = 60  # 例：8 s @ 10 Hz

        self._plan_traj = None  # 一次性预测结果 [N, 3] (x,y,heading)
        self._computed_heading = None  # 缓存计算出的heading [N]
        self._computed_velocity = None  # 缓存计算出的velocity [N, 2]

    def _compute_heading_and_velocity(self):
        """当轨迹不包含heading和velocity时，自动计算并缓存"""
        n_frames = len(self._plan_traj)
        self._computed_heading = np.zeros(n_frames)
        self._computed_velocity = np.zeros((n_frames, 2))

        # 计算每帧的heading和velocity
        for i in range(n_frames):
            # 计算heading（轨迹切线方向）
            if i < n_frames - 1:
                # 用下一帧计算
                dx = self._plan_traj[i + 1, 0] - self._plan_traj[i, 0]
                dy = self._plan_traj[i + 1, 1] - self._plan_traj[i, 1]
            elif i > 0:
                # 最后一帧用前一帧
                dx = self._plan_traj[i, 0] - self._plan_traj[i - 1, 0]
                dy = self._plan_traj[i, 1] - self._plan_traj[i - 1, 1]
            else:
                # 只有一帧时默认heading为0
                dx, dy = 0.0, 1.0  # 默认沿y轴正向

            self._computed_heading[i] = np.arctan2(dy, dx)  # 弧度制

            # 计算velocity（位移/时间）
            if i < n_frames - 1:
                vx = dx / FIXED_DT
                vy = dy / FIXED_DT
            elif i > 0:
                vx = dx / FIXED_DT
                vy = dy / FIXED_DT
            else:
                vx, vy = 0.0, 0.0  # 只有一帧时速度为0

            self._computed_velocity[i] = [vx, vy]

    # ---------- 主入口 ----------
    def act(self, agent_id: str):
        current_step = self.engine.episode_step
        scenario = self.engine.data_manager.current_scenario
        sdc_id = str(scenario["metadata"]["sdc_id"])
        sdc_track = scenario["tracks"][sdc_id]

        # ===== 1. 前 10 帧：完全回放真值 =====
        if current_step <= self.PREDICT_FRAME:
            state = parse_object_state(sdc_track, current_step)
            if state["valid"]:
                self.control_object.set_position(state["position"])
                self.control_object.set_velocity(state["velocity"])
                self.control_object.set_heading_theta(state["heading"])
            return None  # 使用真值，不返回动作

        # ===== 2. 第 10 帧：一次性预测整条未来轨迹 =====
        if self._plan_traj is None:
            plan_traj = self.Sim.run_inference(scenario, 21)
            self._plan_traj = plan_traj[0,:,:2]
        self.engine.agents['default_agent'].plan_traj = self._plan_traj[:, :2]
        # ===== 3. 开环回放：直接拿预存轨迹点 =====
        # 判断轨迹维度，决定是否需要计算heading和velocity
        traj_dim = self._plan_traj.shape[1]
        if traj_dim < 3 :  # 只含xy
            self._compute_heading_and_velocity()

        traj_idx = current_step - self.PREDICT_FRAME - 1
        if traj_idx >= len(self._plan_traj):
            traj_idx = -1  # 超出部分用最后一帧

        target_pos = self._plan_traj[traj_idx, :2]
        if traj_dim >= 5:
            # 轨迹包含heading（维度3或5）
            target_heading = self._plan_traj[traj_idx, 2]
            target_velocity = self._plan_traj[traj_idx, -2:]
        else:
            # 使用计算的heading
            target_heading = self._computed_heading[traj_idx]
            target_velocity = self._computed_velocity[traj_idx]

        # 直接硬设置（无 PID）
        self.control_object.set_position(target_pos)
        self.control_object.set_heading_theta(target_heading)
        self.control_object.set_velocity(target_velocity)

        return None  # 开环不需要输出动作
