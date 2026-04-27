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
        self.cfg = get_final_config(config_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml"))

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
                # Closed-loop planning begins AFTER warmup.
                # We patch the scenario state used for inference so the model sees the executed
                # ego history (closed-loop), but we should start doing this only at the transition.
                if time_index == self.WARMUP_STEPS + 1:
                    # Seed buffer with the last warmup replay frame so that the first inference
                    # has correct ego state at (time_index - 1).
                    last_replay_state = parse_object_state(sdc_track, time_index - 1)
                    if last_replay_state.get("valid", False):
                        v = last_replay_state.get("velocity", None)
                        if v is None or (isinstance(v, (list, tuple, np.ndarray)) and len(v) < 2):
                            # fallback: use scalar speed if provided
                            speed = float(last_replay_state.get("speed", 0.0))
                            heading = float(last_replay_state["heading"])
                            v = np.array([speed * np.cos(heading), speed * np.sin(heading)], dtype=np.float32)
                        self.hist_buffer.clear()
                        self.hist_buffer.append({
                            "position": np.asarray(last_replay_state["position"], dtype=np.float32),
                            "heading": float(last_replay_state["heading"]),
                            "velocity": np.asarray(v, dtype=np.float32),
                            "valid": True,
                        })

                self.current_state = self.update_state(time_index)
                t = time_index - 1
                print("patched_pos:", self.current_state["tracks"][self.sdc_id]["state"]["position"][t, :2],
                      "ego_pos:", self.control_object.position)

                infer_out = self.Sim.run_inference(self.current_state, time_index)
                self._pred_traj = infer_out[0]
                ref_lines = infer_out[1]
                candidates = infer_out[2] if len(infer_out) > 2 else None
                if candidates is not None:
                    self.engine.agents['default_agent'].candidate_trajectories = candidates
                if hasattr(self.engine, 'head_renderer'):
                    self.engine.head_renderer.set_reference_lines(ref_lines)
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

            # ===== RLPlanningPolicy-style tracking (preview point + match point + signed cross-track error) =====
            # 1) 用“预瞄点”而不是车质心做匹配（减少低速抖动/侧向点导致的急转）
            preview_distance = 3.0  # meters, aligned with RLPlanningPolicy.preview_distance
            preview_pos = self._preview_point(current_pos, current_heading, preview_distance)

            # 2) 在轨迹上找离预瞄点最近、且尽量在车头前方的 match 点
            distances = np.linalg.norm(pred_traj[:, :2] - preview_pos, axis=1)
            match_idx = self.get_match_idx(pred_traj, distances, current_pos, current_heading)
            p_match = pred_traj[match_idx, :2]

            # 3) 选取一个沿轨迹“向前”的目标点（lookahead），并确保目标点在车体前方
            lookahead_distance = 3.0  # meters; can be tuned, keep small like RLPlanningPolicy
            target_idx = self._find_lookahead_idx(pred_traj[:, :2], match_idx, lookahead_distance)
            target_idx = self._ensure_forward_target(pred_traj[:, :2], match_idx, target_idx, current_pos, current_heading)
            p_target = pred_traj[target_idx, :2]
            self.engine.agents['default_agent'].key_points = [p_target]

            # 4) 用更远一点的切向算轨迹朝向（减少抖动）
            K_TANGENT = 3
            idx2 = min(match_idx + K_TANGENT, len(pred_traj) - 1)
            p_dir = pred_traj[idx2, :2]
            vec_traj_dir = p_dir - p_match
            if np.linalg.norm(vec_traj_dir) < 1e-3:
                idx_fallback = min(match_idx + 1, len(pred_traj) - 1)
                vec_traj_dir = pred_traj[idx_fallback, :2] - p_match
            if np.linalg.norm(vec_traj_dir) < 1e-3:
                # 极端退化：用车头方向当切向
                vec_traj_dir = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)

            # 5) 横向误差：用“点到轨迹切线”的有符号距离（而非点到目标点距离）
            lateral_error_m = self._signed_cross_track_error(p_target, p_match, vec_traj_dir)
            # Stanley-like scaling (optional). Keep mild to avoid speed->0 blow-up.
            lateral_error = lateral_error_m / max(current_speed, 1.0)

            traj_heading = np.arctan2(vec_traj_dir[1], vec_traj_dir[0])
            heading_error = traj_heading - current_heading
            heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
            combined_error = lateral_error + 2.5 * heading_error


            # 目标速度：从轨迹上采样... (保持不变)
            target_speed_idx = min(match_idx + 5, len(pred_traj) - 1)
            p_target_speed = pred_traj[target_speed_idx, :2]
            target_speed = np.linalg.norm(p_target_speed - p_match) / ((target_speed_idx - match_idx) * FIXED_DT)
            speed_error = current_speed - target_speed

            # --- 3. 使用PID控制器计算最终动作 ---
            # 横向控制现在吃入的是“融合了车头朝向”的修正误差
            steering = self.lateral_pid.get_result(-combined_error)
            # steering = self.lateral_pid.get_result(-1)
            # print(steering)
            throttle_brake = self.longitudinal_pid.get_result(speed_error)
            return [steering, throttle_brake]

    def _preview_point(self, current_pos, current_heading, preview_distance: float) -> np.ndarray:
        """Preview point in world frame, aligned with RLPlanningPolicy.preview_point usage."""
        return np.asarray(current_pos, dtype=np.float32) + preview_distance * np.array(
            [np.cos(current_heading), np.sin(current_heading)], dtype=np.float32
        )

    def _find_lookahead_idx(self, traj_xy: np.ndarray, start_idx: int, lookahead_m: float) -> int:
        """Find an index forward along the trajectory by accumulating arc-length from start_idx."""
        n = len(traj_xy)
        if n == 0:
            return 0
        if n == 1:
            return 0
        start_idx = int(np.clip(start_idx, 0, n - 1))

        acc = 0.0
        i = start_idx
        while i < n - 1 and acc < lookahead_m:
            step = float(np.linalg.norm(traj_xy[i + 1] - traj_xy[i]))
            # protect against duplicate points
            if step < 1e-4:
                step = 0.0
            acc += step
            i += 1
        return i

    def _ensure_forward_target(
        self,
        traj_xy: np.ndarray,
        match_idx: int,
        target_idx: int,
        current_pos: np.ndarray,
        current_heading: float,
        max_extra_steps: int = 20,
    ) -> int:
        """Ensure the chosen target point lies in front of the ego (dot>0); otherwise step forward."""
        n = len(traj_xy)
        if n == 0:
            return 0
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)

        idx = int(np.clip(target_idx, match_idx, n - 1))
        for _ in range(max_extra_steps):
            v = traj_xy[idx] - current_pos
            if float(np.dot(v, heading_vec)) > 0.5:  # require a bit of forward margin
                return idx
            if idx >= n - 1:
                break
            idx += 1
        return idx

    def _signed_cross_track_error(self, p: np.ndarray, p_on_path: np.ndarray, path_tangent: np.ndarray) -> float:
        """Signed lateral distance from point p to the line passing p_on_path with direction path_tangent."""
        t = np.asarray(path_tangent, dtype=np.float32)
        tn = float(np.linalg.norm(t))
        if tn < 1e-6:
            return 0.0
        t = t / tn
        # left normal of tangent
        n = np.array([-t[1], t[0]], dtype=np.float32)
        return float(np.dot(np.asarray(p, dtype=np.float32) - np.asarray(p_on_path, dtype=np.float32), n))

    def update_state(self, time_index):
        """Return a patched scenario state for inference without mutating the engine's scenario.

        We only patch the ego (sdc) track state arrays over the last `replan_frequency` steps
        with the closed-loop executed states, so that the model sees a consistent history
        while the engine's original expert data remains untouched.
        """
        # Base is the current engine scenario (read-only)
        base_state = self.engine.data_manager.current_scenario

        start_idx = time_index - self.replan_frequency   # 向前 replan_frequency 步
        end_idx = time_index  # 当前步（不含）
        length = int(base_state.get('length', 0))

        # During warmup we should not patch; also if nothing to patch, return base_state.
        if time_index <= self.WARMUP_STEPS or length <= 0 or len(self.hist_buffer) == 0:
            return base_state

        # Create a lightweight copy:
        # - Deepcopy metadata/map/etc can be very large; we reuse base_state by reference
        # - But we MUST copy the ego arrays to avoid polluting engine memory.
        patched_state = dict(base_state)
        patched_tracks = dict(base_state.get('tracks', {}))
        patched_state['tracks'] = patched_tracks

        base_track = base_state['tracks'][self.sdc_id]
        patched_track = dict(base_track)
        patched_tracks[self.sdc_id] = patched_track

        base_track_state = base_track['state']
        patched_track_state = dict(base_track_state)
        patched_track['state'] = patched_track_state

        # Copy arrays (owning memory) so in-place assignment won't affect engine.
        patched_traj = np.array(base_track_state['position'], copy=True)
        patched_head = np.array(base_track_state['heading'], copy=True)
        patched_vel = np.array(base_track_state['velocity'], copy=True)

        # We want the buffer tail (most recent executed states) aligned to these timesteps.
        # Example (replan_frequency=1): patch timestep (time_index-1) using hist_buffer[-1].
        steps_to_patch = list(range(start_idx, end_idx))
        k = min(len(steps_to_patch), len(self.hist_buffer))
        if k > 0:
            buf_tail = list(self.hist_buffer)[-k:]
            steps_tail = steps_to_patch[-k:]
            for state, t in zip(buf_tail, steps_tail):
                if 0 <= t < length:
                    patched_traj[t, :2] = state['position']
                    patched_head[t] = state['heading']
                    patched_vel[t] = state['velocity']

        patched_track_state['position'] = patched_traj
        patched_track_state['heading'] = patched_head
        patched_track_state['velocity'] = patched_vel

        return patched_state

    def get_hist_buffer(self, state):
        # 缓存历史
        [current_pos, current_heading, current_speed] = state
        # Convert scalar speed to a 2D velocity vector based on the vehicle heading
        velocity_2d = np.array([current_speed * np.cos(current_heading), current_speed * np.sin(current_heading)])
        state = {
            "position": current_pos,
            "heading": current_heading,
            "velocity": velocity_2d,
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


