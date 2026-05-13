import warnings

import traceback

import os
import pickle
import numpy as np
import h5py

from unitraj.datasets.Pluto_dataset.cost_map_manager import CostMapManager
from unitraj.datasets.Pluto_dataset.utils import save_dict_to_hdf5
from unitraj.datasets.base_dataset import BaseDataset
from scenarionet.common_utils import read_scenario

from shapely.geometry import Polygon
from shapely.geometry import Point, LineString
from typing import List, Tuple
from unitraj.datasets.Pluto_dataset.pluto_utils import  calculate_additional_ego_states,\
    PlutoFeature, interpolate_polyline, _is_lane_like

from unitraj.datasets.Pluto_dataset.map_topology import build_lane_graph, estimate_route_lane_ids




def _get_ego_features(state, ego_category_idx: int = 0, present_idx: int = 20, history_samples: int = 20):
    pos = state['position']
    history_start = max(0, present_idx - history_samples)
    end = present_idx + 1
    
    pos = pos[history_start:end]
    T = len(pos)
    position = pos[..., :2] if pos.shape[-1] >= 2 else pos
    heading = state['heading'][history_start:end]
    vel = state['velocity'][history_start:end]
    velocity = vel[..., :2] if vel.shape[-1] >= 2 else vel

    if 'acceleration' in state:
        accel = state['acceleration'][history_start:end]
        acceleration = accel[..., :2] if accel.shape[-1] >= 2 else accel
    else:
        acceleration = np.zeros((T, 2), dtype=np.float64)

    width = np.array(state['width'][history_start:end]).reshape(-1, 1)
    length = np.array(state['length'][history_start:end]).reshape(-1, 1)
    shape = np.concatenate([width, length], axis=-1)

    valid_mask = state['valid'][history_start:end]
    category = np.array(ego_category_idx, dtype=np.int8)

    return {
        "position": position.astype(np.float64),
        "heading": heading.astype(np.float64),
        "velocity": velocity.astype(np.float64),
        "acceleration": acceleration.astype(np.float64),
        "shape": shape.astype(np.float64),
        "category": category,
        "valid_mask": valid_mask.astype(np.bool_),
    }


def _pad_or_truncate(arr: np.ndarray, target_len: int, axis: int = 0, pad_value: float = 0.0):
    """Pad or truncate a numpy array on a given axis to target_len."""
    arr = np.asarray(arr)
    cur = arr.shape[axis]
    if cur == target_len:
        return arr
    if cur > target_len:
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(0, target_len)
        return arr[tuple(slc)]
    # pad
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target_len - cur)
    return np.pad(arr, pad_width, mode='constant', constant_values=pad_value)


def _box_corners_xy(center_xy: np.ndarray, heading: float, width: float, length: float) -> np.ndarray:
    """Return oriented 2D box corners (4,2) in world frame."""
    center_xy = np.asarray(center_xy, dtype=np.float64).reshape(-1)[:2]
    dx = float(length) / 2.0
    dy = float(width) / 2.0
    corners = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]], dtype=np.float64)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return center_xy[None, :] + corners @ rot.T


class PlutoDataset(BaseDataset):
    """
    Dataset class for Pluto feature format reading NuPlan db files (ScenarioDescription).
    """
    def __init__(self, config=None, is_validation=False):
        # Match pluto_feature_builder.py indexing semantics (no dependency on nuplan types)
        # agent categories: [EGO, VEHICLE, PEDESTRIAN, BICYCLE]
        self.interested_objects_types = ["EGO", "VEHICLE", "PEDESTRIAN", "BICYCLE"]
        # static obstacle categories (nuplan: [CZONE_SIGN, BARRIER, TRAFFIC_CONE, GENERIC_OBJECT])
        self.static_objects_types = ["CZONE_SIGN", "TRAFFIC_BARRIER", "TRAFFIC_CONE", "GENERIC_OBJECT"]
        # map polygon categories (nuplan: [LANE, LANE_CONNECTOR, CROSSWALK])
        self.polygon_types = ["LANE", "LANE_CONNECTOR", "CROSSWALK"]

        self.max_agents = 48
        self.max_static_obstacles = 10
        self.__collate_fn__ = PlutoFeature.collate

        self.radius = config.get('radius', 100)
        self.history_horizon = config.get('history_horizon', 2)
        self.future_horizon = config.get('future_horizon', 8)
        self.sample_interval = config.get('sample_interval', 0.1)
        self.history_samples = int(self.history_horizon / self.sample_interval)
        self.future_samples = int(self.future_horizon / self.sample_interval) - 1
        self.ego_params = None
        super().__init__(config, is_validation)

    def process_data_chunk(self, worker_index):
        with open(os.path.join('tmp', '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f'{worker_index}.h5')

        with h5py.File(hdf5_path, 'w') as f:
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f'{cnt}/{len(data_list)} data processed', flush=True)
                scenario = read_scenario(data_path, mapping, file_name)

                try:
                    output = self.preprocess(scenario)

                    pluto_feature = self.process(output)

                    output = self.postprocess(pluto_feature)

                except Exception as e:
                    print('Warning: {} in {}'.format(e, file_name))
                    traceback.print_exc()
                    output = None

                if output is None: continue

                for i, record in enumerate(output):
                    grp_name = dataset_name + '-' + str(worker_index) + '-' + str(cnt) + '-' + str(i)
                    grp = f.create_group(grp_name)
                    save_dict_to_hdf5(grp, record)

                    file_info = {'h5_path': hdf5_path}
                    file_list[grp_name] = file_info
                del scenario
                del output

        return file_list

    def preprocess(self, scenario, current_step=20):
        """
        In UniTraj, preprocess often generates intermediate dict. 
        Here we skip intermediate formatting and construct pluto feature dict explicitly.
        """
        # We can either return the raw PlutoFeature or dict that HDF5 caches.
        # But HDF5 can't cache PlutoFeature directly if it contains custom objects.
        # Let's return primitive dict.
        self.ego_params = scenario['metadata']['ego_vehicle_parameters']
        all_tracks = scenario['tracks']

        ego_state_list = self.parse_tracks_to_states(all_tracks)
        map_features_list = scenario['map_features']
        traffic_light_status = scenario['dynamic_map_states']
        return [ego_state_list, map_features_list, traffic_light_status, all_tracks]

    def process(self, data, current_step=20):
        [ego_state_list, map_features_list, traffic_light_status, all_tracks] = data
        
        # Start map tracking
        if not hasattr(self, '_lane_graph_cache'):
            self._lane_graph_cache = {}
        
        # Use current_step as the present index instead of hardcoded self.history_samples
        present_idx = current_step
        present_idx = min(present_idx, len(ego_state_list) - 1)
        
        present_ego_state = ego_state_list[present_idx]
        query_xy = present_ego_state['position'][:2]
        # print('query_xy', query_xy)

        # Build lane topology once and reuse in downstream feature extraction.
        # Speed optimization: cache it for the same map instance (object id)
        map_id = id(map_features_list)
        if map_id not in self._lane_graph_cache:
            lane_graph = build_lane_graph(map_features_list, infer_from_geometry=True, geom_link_dist_m=2.0)
            ego_xy = np.asarray([s['position'][:2] for s in ego_state_list], dtype=np.float64)
            route_lane_seq = estimate_route_lane_ids(
                ego_xy,
                lane_graph,
                max_dist_m=1.5,
                min_hold_frames=3,
                no_backtrack=True,
                include_lateral_neighbors=True,
            )
            self._lane_graph_cache[map_id] = (lane_graph, route_lane_seq)
        else:
            lane_graph, route_lane_seq = self._lane_graph_cache[map_id]

        data = {}
        prev_idx = max(0, present_idx - 1)
        data["current_state"] = self._get_ego_current_state(
            ego_state_list[present_idx], ego_state_list[prev_idx]
        )

        ego_features = _get_ego_features(
            all_tracks['ego']['state'],
            ego_category_idx=self.interested_objects_types.index("EGO"),
            present_idx=present_idx,
            history_samples=self.history_samples,
        )
        # print('ego features', ego_features['position'])
        agent_features, agent_tokens, agents_polygon = self._get_agent_features(
            query_xy=query_xy,
            present_idx=present_idx,
            all_tracks=all_tracks
        )

        data["agent"] = {}
        for k in agent_features.keys():
            data["agent"][k] = np.concatenate(
                [ego_features[k][np.newaxis, ...], agent_features[k]], axis=0
            )
        agent_tokens = ["ego"] + agent_tokens

        is_validation = getattr(self, 'is_validation', True)

        if is_validation:
            data["agent_tokens"] = agent_tokens

        data["static_objects"], static_objects = self._get_static_objects_features(
            query_xy=query_xy,
            present_idx=present_idx,
            all_tracks=all_tracks
        )

        data["map"], map_polygon_tokens = self._get_map_features(
            map_features_list=map_features_list,
            query_xy=query_xy,
            route_roadblock_ids=route_lane_seq,
            traffic_light_status=traffic_light_status,
            radius=self.radius,
            present_idx=present_idx,
            lane_graph=lane_graph,
        )

        if not is_validation:
            data["causal"] = self.scenario_casual_reasoning_preprocess(
                ego_features=ego_features,
                agent_features=agent_features,
                agents_tokens=agent_tokens,
                map_polygon_tokens=map_polygon_tokens,
                map_features=data["map"],
                present_idx=self.history_samples,
            )
            data["causal"]["interaction_label"] = self._get_interaction_label(
                ego_features, agent_features
            )
            data["agent"]["valid_mask"][0, self.history_samples + 1:] = data["causal"][
                "fixed_ego_future_valid_mask"
            ]

            cost_map_manager = CostMapManager(
                origin=ego_features["position"][self.history_samples],
                angle=ego_features["heading"][self.history_samples],
                height=600,
                width=600,
                resolution=0.2
            )
            cost_maps_res = cost_map_manager.build_cost_maps(
                # CostMapManager expects static objects with shape info: [x,y,heading,width,length,cat]
                static_objects=static_objects,
                agents=agent_features,
                map_features_list=map_features_list,
                agents_polygon=agents_polygon,
                traffic_light_status=traffic_light_status,
                present_idx=present_idx,
                future_steps=self.future_samples,
                # Pluto default: consider VEHICLE/PEDESTRIAN/BICYCLE as obstacles
                dynamic_obstacle_types=(
                    self.interested_objects_types.index("VEHICLE"),
                    self.interested_objects_types.index("PEDESTRIAN"),
                    self.interested_objects_types.index("BICYCLE"),
                ),
                dynamic_dilation_radius_m=1.0,
            )
            data["cost_maps"] = cost_maps_res["cost_maps"]

        data["reference_line"] = self._get_reference_line_feature(
            ego_features=ego_features,
            map_features_list=map_features_list,
            route_roadblock_ids=route_lane_seq,
            lane_graph=lane_graph,
            training=bool(not self.is_validation)
        )

        return PlutoFeature.normalize(data, first_time=True, radius=self.radius)


    def postprocess(self, pluto_feature):
        return [pluto_feature.data]
        # return pluto_feature

    # def __getitem__(self, idx):
    #     # We fetch the dict from parent class HDF5 cache,
    #     # and then wrap it in PlutoFeature at iteration time!
    #     record = super().__getitem__(idx)
    #     # Reconstruct structured dict
    #     data = {
    #         'agent': {
    #             'position': record['agent/position'],
    #             'heading': record['agent/heading'],
    #             'velocity': record['agent/velocity'],
    #             'shape': record['agent/shape'],
    #             'category': record['agent/category'],
    #             'valid_mask': record['agent/valid_mask'],
    #         },
    #         'map': {
    #             'point_position': record['map/point_position'],
    #             'point_vector': record['map/point_vector'],
    #             'polygon_center': record['map/polygon_center'],
    #             'polygon_type': record['map/polygon_type'],
    #             'valid_mask': record['map/valid_mask']
    #         },
    #         'current_state': record['current_state']
    #     }
    #     return PlutoFeature(data=data)

    def parse_tracks_to_states(self, tracks):
        """
        参数:
            scenario (dict): 包含 'tracks' 键的字典，'tracks' 是一个字典，键为对象ID，值为包含 'state' 的字典。
                             'state' 包含 'position', 'heading', 'velocity', 'valid', 'length', 'width', 'height' 等字段。
                             每个字段都是形状为 (T, ...) 的数组，其中 T 是时间帧数。

        返回:
            ego_state_list (list): 自车状态列表，按帧排列，每个元素是该帧自车的状态字典。
            tracked_objects_list (list): 周围车辆状态列表，按帧排列，每个元素是该帧所有周围车辆的状态字典列表。
                                        结构: [ [frame_0_ego_state], [frame_1_ego_state], ... ]
                                              和 [ [frame_0_obj1_state, frame_0_obj2_state, ...], [frame_1_obj1_state, frame_1_obj2_state, ...], ... ]
        """
        ego_state = tracks['ego']['state']
        T = len(ego_state['position'])
        ego_state_list = []

        # 逐帧提取状态
        for frame_idx in range(T):
            # 处理自车状态
            ego_frame_state = {
                'position': ego_state['position'][frame_idx].tolist(),
                'heading': ego_state['heading'][frame_idx].item(),
                'velocity': ego_state['velocity'][frame_idx].tolist(),
                'valid': ego_state['valid'][frame_idx].item(),
                'length': ego_state['length'][frame_idx].item(),
                'width': ego_state['width'][frame_idx].item(),
                'height': ego_state['height'][frame_idx].item()
            }
            ego_state_list.append(ego_frame_state)
        return ego_state_list

    def _get_ego_current_state(self, ego_state, prev_state):
        state = np.zeros(7, dtype=np.float64)
        state[0:2] = ego_state['position'][:2]
        state[2] = ego_state['heading']
        state[3:5] = ego_state['velocity']

        steering_angle, yaw_rate = calculate_additional_ego_states(
            ego_state, prev_state, self.ego_params
        )

        state[5] = steering_angle
        state[6] = yaw_rate

        return state

    def _get_agent_features(
        self,
        query_xy,
        present_idx: int,
        all_tracks,
    ):
        # Find valid non-ego agents at present_idx
        present_agents = []
        for obj_id, track in all_tracks.items():
            if str(obj_id) == 'ego' or str(track.get("type", "")).upper() not in self.interested_objects_types: # rough check for ego
                continue
            state = track['state']
            if state['valid'][present_idx]:
                pos = state['position'][present_idx][:2]
                dist = np.linalg.norm(np.array(pos) - np.array(query_xy))
                present_agents.append((dist, obj_id, track))
                
        present_agents.sort(key=lambda x: x[0])
        present_agents = present_agents[:self.max_agents]
        
        N, T = min(len(present_agents), self.max_agents), self.history_samples + 1
        
        position = np.zeros((N, T, 2), dtype=np.float64)
        heading = np.zeros((N, T), dtype=np.float64)
        velocity = np.zeros((N, T, 2), dtype=np.float64)
        shape = np.zeros((N, T, 2), dtype=np.float64)
        category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, T), dtype=np.bool_)
        polygon = [None] * N
        
        agent_tokens = []

        if N == 0:
            return (
                {
                    "position": position,
                    "heading": heading,
                    "velocity": velocity,
                    "shape": shape,
                    "category": category,
                    "valid_mask": valid_mask,
                },
                [],
                [],
            )

        for idx, (_dist, obj_id, track) in enumerate(present_agents):
            agent_tokens.append(obj_id)
            track_state = track['state']

            # Minimal, fast path: slice/pad to fixed horizon.
            history_start = max(0, present_idx - self.history_samples)
            end = present_idx + 1

            pos_seq = np.asarray(track_state['position'], dtype=np.float64)[history_start:end, :2]
            vel_seq = np.asarray(track_state['velocity'], dtype=np.float64)[history_start:end, :2]
            hdg_seq = np.asarray(track_state['heading'], dtype=np.float64)[history_start:end]
            vld_seq = np.asarray(track_state['valid'], dtype=bool)[history_start:end]

            position[idx] = pos_seq
            velocity[idx] = vel_seq
            heading[idx] = hdg_seq
            valid_mask[idx] = vld_seq

            # width/length: keep constant, expand to all timesteps
            w0 = float(np.asarray(track_state.get('width', 0.0)).reshape(-1)[0])
            l0 = float(np.asarray(track_state.get('length', 0.0)).reshape(-1)[0])
            shape[idx, :, 0] = w0
            shape[idx, :, 1] = l0
            
            track_type = str(track.get('type', '')).upper()
            if track_type in {"VEHICLE", "CAR", "TRUCK", "BUS"}:
                cat = self.interested_objects_types.index("VEHICLE")
            elif track_type in {"PEDESTRIAN", "PEDESTRIAN_ADULT", "PEDESTRIAN_CHILD"}:
                cat = self.interested_objects_types.index("PEDESTRIAN")
            elif track_type in {"BICYCLE", "BICYCLIST", "CYCLIST"}:
                cat = self.interested_objects_types.index("BICYCLE")
            else:
                # fallback: treat unknown dynamic agent as VEHICLE
                cat = self.interested_objects_types.index("VEHICLE")
            category[idx] = cat

            # Build present-time polygon for this agent (used by cost map parked-agent logic).
            center_now = pos_seq[self.history_samples]
            heading_now = float(hdg_seq[self.history_samples])
            poly_xy = _box_corners_xy(center_now, heading_now, float(w0), float(l0))
            polygon[idx] = Polygon(poly_xy)
            
        agent_features = {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }

        return agent_features, agent_tokens, polygon

    def _get_static_objects_features(
        self,
        query_xy,
        present_idx: int,
        all_tracks,
    ):
        static_objects = []
        # dynamic_types = {'VEHICLE', 'PEDESTRIAN', 'BICYCLE'}

        for obj_id, track in all_tracks.items():
            track_type = str(track.get('type', '')).upper()
            if track_type in self.interested_objects_types:
                continue

            state = track['state']
            if not state['valid'][present_idx]:
                continue

            pos = state['position'][present_idx][:2]
            if np.linalg.norm(np.array(pos) - np.array(query_xy)) > self.radius:
                continue

            heading = state['heading'][present_idx]
            length = state['length'][present_idx] if isinstance(state['length'], (list, np.ndarray)) else state[
                'length']
            width = state['width'][present_idx] if isinstance(state['width'], (list, np.ndarray)) else state[
                'width']

            # Map static obstacle type to pluto_feature_builder indices
            if "BARRIER" in track_type:
                cat = self.static_objects_types.index("TRAFFIC_BARRIER")
            elif "CONE" in track_type or "TRAFFIC_CONE" in track_type:
                cat = self.static_objects_types.index("TRAFFIC_CONE")
            elif "SIGN" in track_type or "CZONE_SIGN" in track_type:
                cat = self.static_objects_types.index("CZONE_SIGN")
            else:
                cat = self.static_objects_types.index("GENERIC_OBJECT")
            static_objects.append([pos[0], pos[1], heading,float(width) , float(length), cat])

        if len(static_objects) > 0:
            static_objects = np.array(static_objects, dtype=np.float64)
            valid_mask = np.ones(len(static_objects), dtype=np.bool_)
        else:
            static_objects = np.zeros((0, 6), dtype=np.float64)
            valid_mask = np.zeros(0, dtype=np.bool_)

        return {
            "position": static_objects[:, :2],
            "heading": static_objects[:, 2],
            "shape": static_objects[:, 3:5],
            "category": static_objects[:, -1].astype(np.int8),
            "valid_mask": valid_mask,
        }, static_objects

    def _get_map_features(
        self,
        map_features_list: dict,
        query_xy,
        route_roadblock_ids: list,
        traffic_light_status,
        radius: float,
        sample_points: int = 20,
        present_idx: int = None,
        lane_graph=None,
    ):
        present_idx_use = present_idx if present_idx is not None else self.history_samples
        route_ids = set(str(route_id) for route_id in route_roadblock_ids)

        # traffic_light_status (SD) is typically a dict keyed by lane_connector_id/lane_id.
        # We convert it to lane_id -> numeric status at present_idx.
        # IMPORTANT: Align with NuPlan TrafficLightStatusType used by Pluto pretrained weights:
        #   GREEN=0, YELLOW=1, RED=2, UNKNOWN=3
        state_mapping = {
            "TRAFFIC_LIGHT_GREEN": 0,
            "TRAFFIC_LIGHT_YELLOW": 1,
            "TRAFFIC_LIGHT_RED": 2,
            "TRAFFIC_LIGHT_UNKNOWN": 3,
        }

        def _tl_to_int(x) -> int:
            if x is None:
                return 0
            s = str(x).upper()
            return int(state_mapping.get(s, 3))

        tls = {}
        for lane_id, tl_info in (traffic_light_status or {}).items():
            state = tl_info.get('state', {}) if isinstance(tl_info.get('state', None), dict) else {}
            obj_state = state.get('object_state', None)
            tls[str(lane_id)] = _tl_to_int(obj_state[present_idx_use])

        lane_objects = []
        crosswalk_objects = []
        query_pos = query_xy

        for map_id, map_feat in map_features_list.items():
            # Check distance
            polyline = map_feat.get('polyline', None)
            polygon = map_feat.get('polygon', None)
            
            pts = polyline if polyline is not None and len(polyline) > 0 else polygon
            if pts is None or len(pts) == 0:
                continue
                
            center = np.mean(pts[:, :2], axis=0)
            if np.linalg.norm(center - query_pos) > radius:
                continue

            obj_type = str(map_feat.get('type', '')).upper()
            if 'CROSSWALK' in obj_type:
                crosswalk_objects.append((map_id, map_feat))
            elif _is_lane_like(obj_type) :
                lane_objects.append((map_id, map_feat))

        # Build per-polygon records first to avoid shape/broadcast issues when
        # some polygons are invalid and must be skipped.
        P = sample_points
        rec_point_position = []
        rec_point_vector = []
        rec_point_side = []
        rec_point_orientation = []
        rec_polygon_center = []
        rec_polygon_position = []
        rec_polygon_orientation = []
        rec_polygon_type = []
        rec_polygon_on_route = []
        rec_polygon_tl_status = []
        rec_polygon_speed_limit = []
        rec_polygon_has_speed_limit = []
        rec_polygon_road_block_id = []

        kept_object_ids = []

        for map_id, lane in lane_objects:
            polyline = lane.get('polyline', np.zeros((2, 2)))
            if len(polyline) < 2:
                polyline = np.concatenate([polyline, np.zeros((2 - len(polyline), polyline.shape[-1]))])
                
            # Sample discrete path from polyline
            centerline = interpolate_polyline(polyline[:, :2], sample_points + 1)

            # Approximate lane boundaries.
            # 1) Prefer neighbor lane centerlines (left/right) if available; this matches your unified schema
            #    where left/right neighbors are lane IDs.
            # 2) Else, try to estimate a lateral offset from polygon (lane boundary) if provided.
            # 3) Fallback: duplicate centerline.
            left_bound = None
            right_bound = None

            try:
                lid = int(map_id) if str(map_id).isdigit() else None
            except Exception:
                lid = None

            if lid is not None and lid in lane_graph.lanes:
                node = lane_graph.lanes[lid]
                if node.left_neighbors:
                    nb = node.left_neighbors[0]
                    if nb in lane_graph.lanes:
                        left_bound = interpolate_polyline(lane_graph.lanes[nb].centerline, sample_points + 1)
                if node.right_neighbors:
                    nb = node.right_neighbors[0]
                    if nb in lane_graph.lanes:
                        right_bound = interpolate_polyline(lane_graph.lanes[nb].centerline, sample_points + 1)

            if left_bound is None or right_bound is None:
                polygon = lane.get('polygon', None)
                poly_xy = None
                if polygon is not None:
                    try:
                        poly_xy = np.asarray(polygon, dtype=np.float64)
                        if poly_xy.ndim == 2 and poly_xy.shape[0] >= 3:
                            poly_xy = poly_xy[:, :2]
                        else:
                            poly_xy = None
                    except Exception:
                        poly_xy = None

                if poly_xy is not None:
                    # Estimate lane half-width as median distance from centerline samples to polygon vertices.
                    # This is a rough proxy but better than duplicating the centerline.
                    d = np.min(np.linalg.norm(poly_xy[None, :, :] - centerline[:, None, :], axis=-1), axis=1)
                    half_w = float(np.clip(np.median(d), 0.5, 6.0))
                    # Build left/right offset using local tangent normals.
                    tang = np.diff(centerline, axis=0, prepend=centerline[0:1])
                    nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
                    nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
                    nrm = nrm / np.clip(nrm_norm, 1e-6, None)
                    if left_bound is None:
                        left_bound = centerline + half_w * nrm
                    if right_bound is None:
                        right_bound = centerline - half_w * nrm

            edges = np.stack([centerline, left_bound, right_bound], axis=0)

            vec = edges[:, 1:] - edges[:, :-1]
            pos = edges[:, :-1]
            ori = np.arctan2(vec[:, :, 1], vec[:, :, 0])

            rec_point_vector.append(vec)
            rec_point_position.append(pos)
            rec_point_orientation.append(ori)
            rec_point_side.append(np.arange(3, dtype=np.int8))

            rec_polygon_center.append(
                np.concatenate(
                    [
                        centerline[int(sample_points / 2)],
                        [ori[0, int(sample_points / 2)]],
                    ],
                    axis=-1,
                )
            )
            rec_polygon_position.append(centerline[0])
            rec_polygon_orientation.append(float(ori[0, 0]))
            # - MetaDriveType.LANE_SURFACE_STREET for ROADBLOCK interior edges (treat as LANE)
            # - MetaDriveType.LANE_SURFACE_UNSTRUCTURE for ROADBLOCK_CONNECTOR interior edges (treat as LANE_CONNECTOR)
            lane_type = str(lane.get('type', '')).upper()
            if "LANE_SURFACE_UNSTRUCTURE" in lane_type:
                rec_polygon_type.append(int(self.polygon_types.index("LANE_CONNECTOR")))
            else:
                rec_polygon_type.append(int(self.polygon_types.index("LANE")))
            rec_polygon_on_route.append(bool(str(map_id) in route_ids))
            rec_polygon_tl_status.append(int(tls.get(str(map_id), 0)))
            rec_polygon_has_speed_limit.append(False)
            rec_polygon_speed_limit.append(0.0)
            rec_polygon_road_block_id.append(int(map_id))

            kept_object_ids.append(map_id)

        for map_id, crosswalk in crosswalk_objects:
            polygon = crosswalk.get('polygon', np.zeros((4, 2)))
            if len(polygon) < 2:
                continue
                
            # Treat crosswalk polygon as edges
            edges = np.tile(interpolate_polyline(polygon[:, :2], sample_points + 1)[None, :], (3, 1, 1))

            vec = edges[:, 1:] - edges[:, :-1]
            pos = edges[:, :-1]
            ori = np.arctan2(vec[:, :, 1], vec[:, :, 0])

            rec_point_vector.append(vec)
            rec_point_position.append(pos)
            rec_point_orientation.append(ori)
            rec_point_side.append(np.arange(3, dtype=np.int8))
            rec_polygon_center.append(
                np.concatenate(
                    [
                        edges[0, int(sample_points / 2)],
                        [ori[0, int(sample_points / 2)]],
                    ],
                    axis=-1,
                )
            )
            rec_polygon_position.append(edges[0, 0])
            rec_polygon_orientation.append(float(ori[0, 0]))
            rec_polygon_type.append(int(self.polygon_types.index("CROSSWALK")))
            rec_polygon_on_route.append(False)
            rec_polygon_tl_status.append(0)
            rec_polygon_has_speed_limit.append(False)
            rec_polygon_speed_limit.append(0.0)
            rec_polygon_road_block_id.append(int(map_id))

            kept_object_ids.append(map_id)

        # Stack records
        M = len(rec_point_position)
        point_position = np.stack(rec_point_position, axis=0) if M > 0 else np.zeros((0, 3, P, 2), dtype=np.float64)
        point_vector = np.stack(rec_point_vector, axis=0) if M > 0 else np.zeros((0, 3, P, 2), dtype=np.float64)
        point_orientation = np.stack(rec_point_orientation, axis=0) if M > 0 else np.zeros((0, 3), dtype=np.float64)
        point_side = np.stack(rec_point_side, axis=0) if M > 0 else np.zeros((0, 3), dtype=np.int8)
        polygon_center = np.stack(rec_polygon_center, axis=0) if M > 0 else np.zeros((0, 3), dtype=np.float64)
        polygon_position = np.stack(rec_polygon_position, axis=0) if M > 0 else np.zeros((0, 2), dtype=np.float64)
        polygon_orientation = np.asarray(rec_polygon_orientation, dtype=np.float64) if M > 0 else np.zeros((0,), dtype=np.float64)
        polygon_type = np.asarray(rec_polygon_type, dtype=np.int8) if M > 0 else np.zeros((0,), dtype=np.int8)
        polygon_on_route = np.asarray(rec_polygon_on_route, dtype=np.bool_) if M > 0 else np.zeros((0,), dtype=np.bool_)
        polygon_tl_status = np.asarray(rec_polygon_tl_status, dtype=np.int8) if M > 0 else np.zeros((0,), dtype=np.int8)
        polygon_speed_limit = np.asarray(rec_polygon_speed_limit, dtype=np.float64) if M > 0 else np.zeros((0,), dtype=np.float64)
        polygon_has_speed_limit = np.asarray(rec_polygon_has_speed_limit, dtype=np.bool_) if M > 0 else np.zeros((0,), dtype=np.bool_)
        polygon_road_block_id = np.asarray(rec_polygon_road_block_id, dtype=np.int32) if M > 0 else np.zeros((0,), dtype=np.int32)

        object_ids = kept_object_ids

        map_features = {
            "point_position": point_position,
            "point_vector": point_vector,
            "point_orientation": point_orientation,
            "point_side": point_side,
            "polygon_center": polygon_center,
            "polygon_position": polygon_position,
            "polygon_orientation": polygon_orientation,
            "polygon_type": polygon_type,
            "polygon_on_route": polygon_on_route,
            "polygon_tl_status": polygon_tl_status,
            "polygon_has_speed_limit": polygon_has_speed_limit,
            "polygon_speed_limit": polygon_speed_limit,
            "polygon_road_block_id": polygon_road_block_id,
            # allow causal reasoning to access raw polygons by map_id
            "_raw_map_features": map_features_list,
        }

        return map_features, object_ids

    def scenario_casual_reasoning_preprocess(
        self,
        ego_features,
        agent_features,
        agents_tokens,
        map_polygon_tokens,
        map_features=None,
        present_idx=None,
    ):
        """Heuristic causal reasoning without nuPlan ScenarioManager.

        找到自车前方最近的动态车辆，识别对自车有影响的红灯区域，计算自车在考虑前方障碍物和红灯后的可行驶路径，判断自车未来轨迹是否会进入红灯区域

        Limitations vs nuplan:
        - no drivable-area query
        - no lane-graph-based leading-object inference
        - no precise stop-line occupancy; we use map polygon containment as proxy
        """
        if present_idx is None:
            present_idx = self.history_samples

        num_agents = len(agents_tokens)
        num_maps = len(map_polygon_tokens)
        T_future = self.future_samples

        leading_agent_mask = np.zeros(num_agents, dtype=bool)
        leading_distance = np.zeros(num_agents, dtype=np.float64)
        ego_care_red_light_mask = np.zeros(num_maps, dtype=bool)
        fixed_ego_future_valid_mask = np.ones(T_future, dtype=bool)

        # present ego pose
        ego_pos = ego_features["position"][present_idx]
        ego_heading = float(ego_features["heading"][present_idx])
        ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)

        # --- leading dynamic agents (forward cone in ego heading) ---
        nearest_leading_agent_idx = None
        nearest_leading_agent_dist = None

        if agent_features is not None and agent_features["position"].shape[0] > 0:
            # agent_features are non-ego agents only, aligned with agents_tokens[1:]
            pos_now = agent_features["position"][:, present_idx]
            valid_now = agent_features["valid_mask"][:, present_idx]
            rel = pos_now - ego_pos[None, :]
            forward = rel @ ego_dir  # projection
            lateral = np.abs(rel[:, 0] * (-ego_dir[1]) + rel[:, 1] * ego_dir[0])

            # candidates in front within lateral band
            cand = valid_now & (forward > 0.0) & (lateral < 3.5)
            if cand.any():
                # sort by forward distance
                order = np.argsort(forward + (~cand) * 1e6)
                for j in order:
                    if not cand[j]:
                        continue
                    # mark this as leading
                    idx_token = j + 1  # shift because ego at 0 in agents_tokens
                    leading_agent_mask[idx_token] = True
                    leading_distance[idx_token] = float(forward[j])
                    if nearest_leading_agent_idx is None:
                        nearest_leading_agent_idx = idx_token
                        nearest_leading_agent_dist = float(forward[j])
                    # also mark other close-in-front agents as leading
                    if forward[j] < 30.0:
                        continue
                    break

        # --- red light polygons ---
        nearest_red_poly = None
        nearest_red_poly_dist = None

        def _is_red(x):
            # numeric encoding: 3 means RED
            if isinstance(x, (np.integer, int)):
                return int(x) == 3
            if isinstance(x, (bytes, str)):
                return "RED" in str(x).upper()
            return False

        if map_features is not None and "polygon_tl_status" in map_features:
            tl_status = map_features["polygon_tl_status"]
            poly_pos = map_features.get("polygon_position", None)
            for i in range(min(num_maps, len(tl_status))):
                if _is_red(tl_status[i]):
                    ego_care_red_light_mask[i] = True
                    if poly_pos is not None:
                        d = float(np.linalg.norm(poly_pos[i] - ego_pos))
                        if nearest_red_poly_dist is None or d < nearest_red_poly_dist:
                            nearest_red_poly_dist = d
                            nearest_red_poly = i

        # nuplan builder: "waiting for red light without lead" means nearest leading object is red light
        is_waiting_for_red_light_without_lead = bool(nearest_red_poly is not None and nearest_leading_agent_idx is None)

        # future valid mask: if ego future enters nearest red polygon, invalidate remaining future
        if nearest_red_poly is not None and map_features is not None:
            token = map_polygon_tokens[nearest_red_poly] if nearest_red_poly < len(map_polygon_tokens) else None
            poly = None
            raw_map = map_features.get("_raw_map_features", None)
            if isinstance(raw_map, dict) and token is not None:
                raw = raw_map.get(str(token), raw_map.get(token))
                if isinstance(raw, dict):
                    poly = raw.get("polygon", None)

            if poly is not None and len(poly) >= 3:
                shp = Polygon(poly[:, :2])
                future_pos = ego_features["position"][present_idx + 1 : present_idx + 1 + T_future][:,:2]
                for i in range(min(T_future, len(future_pos))):
                    pt = future_pos[i]
                    if shp.contains(Polygon([pt, pt, pt]).centroid):
                        fixed_ego_future_valid_mask[i:] = False
                        break

        # free path points along ego heading, with end limited by nearest lead and red light
        ego_speed = float(np.linalg.norm(ego_features["velocity"][present_idx])) if "velocity" in ego_features else 0.0
        free_path_start = ego_speed**2 / (2 * 5.0) + 2.5
        free_path_end = max(7.0, ego_speed**2 / (2 * 1.5))
        if nearest_leading_agent_dist is not None:
            free_path_end = min(free_path_end, nearest_leading_agent_dist)
        if nearest_red_poly_dist is not None:
            free_path_end = min(free_path_end, nearest_red_poly_dist)

        if free_path_end <= free_path_start:
            free_path_points = np.zeros((0, 3), dtype=np.float64)
            free_path_points_angle = np.zeros((0,), dtype=np.float64)
        else:
            n = max(int((free_path_end - free_path_start) / 1.0), 2)
            ds = np.linspace(free_path_start + 3.0, max(free_path_start + 3.0, free_path_end - 3.0), n)
            points = (ego_pos[None, :] + ds[:, None] * ego_dir[None, :]).astype(np.float64)
            headings = np.full((points.shape[0], 1), ego_heading, dtype=np.float64)
            free_path_points = np.hstack([points, headings])

        return {
            "is_waiting_for_red_light_without_lead": is_waiting_for_red_light_without_lead,
            "leading_agent_mask": leading_agent_mask,
            "leading_distance": leading_distance,
            "ego_care_red_light_mask": ego_care_red_light_mask,
            "fixed_ego_future_valid_mask": fixed_ego_future_valid_mask,
            "free_path_points": free_path_points,
        }

    def _get_interaction_label(self, ego, agents):
        """Compute interaction label between ego and each agent.

        This is a dependency-free approximation of PlutoFeatureBuilder's interaction label.
        We:
        1) Find nearest ego-agent distance over future horizon (t > history_samples)
        2) If below threshold and boxes intersect, mark as interaction
        3) Label is time difference (ego_t - agent_t) in the closest pair, clipped, with 0 meaning no interaction.

        Output shape matches builder usage: (N_agents + 1,) including ego at index 0.
        """
        start = self.history_samples + 1
        ego_heading = ego["heading"][start:]
        ego_position = ego["position"][start:]

        agents_position = agents["position"][:, start:]
        agents_heading = agents["heading"][:, start:]
        agents_shape = agents["shape"][:, start:]
        agents_valid = agents["valid_mask"][:, start:]

        if agents_position.shape[0] == 0 or agents_position.shape[1] == 0 or ego_position.shape[0] == 0:
            return np.zeros(1, dtype=np.int64)

        N, T = agents_position.shape[:2]
        Te = ego_position.shape[0]

        # pairwise distances for each agent across time pairs (agent_t, ego_t)
        # Build (N, T, Te) distances in numpy for speed/memory constraints: do incremental min
        min_dist = np.full((N,), 1e9, dtype=np.float64)
        min_idx = np.full((N,), -1, dtype=np.int64)

        for i in range(N):
            if not agents_valid[i].any():
                continue
            # restrict to valid timesteps
            valid_ts = np.where(agents_valid[i])[0]
            if len(valid_ts) == 0:
                continue
            # compute cdist for valid agent steps to all ego steps
            a = agents_position[i, valid_ts]  # (Tv,2)
            e = ego_position  # (Te,2)
            d = np.linalg.norm(a[:, None, :] - e[None, :, :], axis=-1)  # (Tv,Te)
            flat = d.reshape(-1)
            j = int(flat.argmin())
            md = float(flat[j])
            if md < min_dist[i]:
                min_dist[i] = md
                # encode back to agent_t, ego_t in full-T coordinates
                agent_t = int(valid_ts[j // Te])
                ego_t = int(j % Te)
                min_idx[i] = agent_t * Te + ego_t

        interact_flag = min_dist < 4.0

        # collision check with oriented boxes
        for i in range(N):
            if not interact_flag[i] or min_idx[i] < 0:
                continue
            agent_t = int(min_idx[i] // Te)
            ego_t = int(min_idx[i] % Te)
            agent_shape = agents_shape[i, agent_t]
            agent_box = self._build_agent_bbox(
                agents_position[i, agent_t],
                agents_heading[i, agent_t],
                float(agent_shape[0]),
                float(agent_shape[1]),
            )
            ego_box = self._build_ego_bbox(ego_position[ego_t], float(ego_heading[ego_t]))
            if not agent_box.intersects(ego_box):
                interact_flag[i] = False

        # label
        interact_label = np.zeros((N,), dtype=np.int64)
        for i in range(N):
            if not interact_flag[i] or min_idx[i] < 0:
                continue
            agent_t = int(min_idx[i] // Te)
            ego_t = int(min_idx[i] % Te)
            interact_label[i] = ego_t - agent_t

        # prepend ego
        return np.concatenate([np.zeros(1, dtype=np.int64), interact_label])

    @staticmethod
    def _get_interact_type(index, T=80):
        row, col = index // T, index % T
        if row == col:
            return 0  # collision or self
        return col - row

    def _build_agent_bbox(self, xy, angle, width, length):
        dx = length / 2
        dy = width / 2
        corners = np.array([
            [dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]
        ])
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        center = xy[:2] if (isinstance(xy, np.ndarray) and xy.shape[-1] >= 2) else xy
        return Polygon(center + corners @ rot.T)

    def _build_ego_bbox(self, xy, angle):
        center = xy + 1.67 * np.array([np.cos(angle), np.sin(angle)])
        width = getattr(self, 'width', 2.0)
        length = getattr(self, 'length', 5.0)
        return self._build_agent_bbox(center, angle, width, length)

    def _get_ego_head_position(self, xy, angle):
        ego_len = getattr(self, 'length', 5.0)
        return xy + ego_len * np.array([np.cos(angle), np.sin(angle)]) / 2


    def _get_reference_line_feature(
        self,
        ego_features,
        map_features_list,
        route_roadblock_ids=None,
        lane_graph=None,
        training=False,
    ):
        ego_pos = ego_features["position"][-1]
        ego_heading = ego_features["heading"][-1]

        radius = self.radius

        # route_roadblock_ids here is actually route lane-id sequence (forward ordered).
        route_seq = [int(x) for x in (route_roadblock_ids or [])]

        def wrap_to_pi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        # Build a working lane set within radius to avoid excessive computation.
        # If route is available, restrict to route lanes (plus a small neighbor set) to reduce branching.
        lane_ids = []
        lane_polys = {}
        lane_xyz = {}
        from typing import Set
        neighbor_extra: Set[int] = set()

        if route_seq:
            # Expand route_set with immediate left/right neighbors to tolerate minor map mismatches.
            for lid in route_seq:
                node = lane_graph.lanes.get(lid)
                if node is None:
                    continue
                neighbor_extra.update(node.left_neighbors)
                neighbor_extra.update(node.right_neighbors)

        allowed_lanes = None
        if route_seq:
            # Keep a deterministic iteration order to avoid subtle randomness in candidate selection.
            # We prioritize lanes on the forward route sequence, then add nearby neighbor lanes.
            allowed_lanes = set(route_seq) | neighbor_extra
            allowed_lanes_ordered = []
            seen = set()
            for lid in route_seq:
                if lid in allowed_lanes and lid not in seen:
                    allowed_lanes_ordered.append(lid)
                    seen.add(lid)
            for lid in sorted(allowed_lanes):
                if lid not in seen:
                    allowed_lanes_ordered.append(lid)
                    seen.add(lid)
        else:
            allowed_lanes_ordered = None

        # Prefer direct lookup in allowed_lanes (route corridor) to avoid scanning the whole lane graph.
        # Fall back to full scan only when route is unavailable.
        iter_lids = allowed_lanes_ordered if allowed_lanes_ordered is not None else list(lane_graph.lanes.keys())
        for lid in iter_lids:
            node = lane_graph.lanes.get(lid)
            if node is None:
                continue
            cl = node.centerline
            if cl is None or len(cl) < 2:
                continue
            # quick radius filter by nearest centerline point
            min_dist = float(np.min(np.linalg.norm(cl - ego_pos[None, :2], axis=1)))
            if min_dist > radius:
                continue
            lane_ids.append(lid)
            lane_polys[lid] = cl
            lane_xyz[lid] = node

        # Find candidate start lanes near ego, aligned with ego heading and in front.
        candidates: List[Tuple[int, float]] = []
        ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)
        DIST_TH = 8.0
        HEADING_TH = 1.05
        FORWARD_TH = -2.0

        for lid in lane_ids:
            poly = lane_polys[lid]
            dists = np.linalg.norm(poly - ego_pos[None, :2], axis=1)
            k = int(np.argmin(dists))
            dist = float(dists[k])
            if dist > DIST_TH:
                continue
            # local tangent
            if k == 0:
                p0, p1 = poly[0], poly[1]
            elif k >= len(poly) - 1:
                p0, p1 = poly[-2], poly[-1]
            else:
                p0, p1 = poly[k - 1], poly[k + 1]
            traj_vec = (p1 - p0).astype(np.float64)
            n = float(np.linalg.norm(traj_vec))
            if n < 1e-3:
                continue
            traj_heading = float(np.arctan2(traj_vec[1], traj_vec[0]))
            heading_diff = abs(wrap_to_pi(traj_heading - ego_heading))
            if heading_diff > HEADING_TH:
                continue
            vec_near = (poly[k] - ego_pos[:2]).astype(np.float64)
            forward = float(np.dot(vec_near, ego_dir))
            if forward <= FORWARD_TH:
                continue
            candidates.append((lid, dist))

        # Trim route_seq to only keep lane ids from ego current lane onward.
        if route_seq and candidates:
            start_idx = None
            for lid, _ in sorted(candidates, key=lambda x: x[1]):
                if lid in route_seq:
                    start_idx = route_seq.index(lid)
                    break
            # ======== 👇 解决隐患3 start_idx is None的新增代码 👇 ========
            if start_idx is None:
                best_dist = float('inf')
                best_idx = 0
                for i, r_lid in enumerate(route_seq):
                    # 找到目前 route_seq 里离主车最近的车道
                    node = lane_graph.lanes.get(r_lid)
                    if node and node.centerline is not None and len(node.centerline) > 0:
                        dist = float(np.min(np.linalg.norm(node.centerline - ego_pos[None, :2], axis=1)))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
                start_idx = best_idx
            # ======== 👆 新增代码结束 👆 ========
            if start_idx is not None:
                route_seq = route_seq[start_idx:]
        route_set = set(route_seq)

        if len(candidates) == 0:
            print('—————————————— Warning,参考线为直线————————————')
            xs = ego_pos[0] + np.linspace(0, radius, int(radius)) * np.cos(ego_heading)
            ys = ego_pos[1] + np.linspace(0, radius, int(radius)) * np.sin(ego_heading)
            fake_line = np.stack([xs, ys, np.full_like(xs, ego_heading)], axis=-1)
            reference_lines = [fake_line]
            reference_line_route_ids = [route_seq[0] if route_seq else -1]
        else:
            # choose nearest few as starting points
            # If route is available, prefer start lanes that are exactly on route.
            cand_sorted = sorted(candidates, key=lambda x: x[1])
            start_lanes_on_route = [lid for lid, _ in cand_sorted if (not route_seq) or (lid in route_set)]
            start_lanes = start_lanes_on_route[:6] if start_lanes_on_route else [lid for lid, _ in cand_sorted[:6]]

            # Add immediate left/right neighbors to increase candidate diversity (parallel lanes).
            # Keep it bounded to avoid combinatorial explosion.
            extra_starts = []
            for lid in list(start_lanes):
                node = lane_graph.lanes.get(lid)
                if node is None:
                    continue
                for nb in (node.left_neighbors + node.right_neighbors):
                    if allowed_lanes is not None and nb not in allowed_lanes:
                        continue
                    extra_starts.append(nb)
            # unique preserve order
            seen = set(start_lanes)
            for nb in extra_starts:
                if nb not in seen:
                    start_lanes.append(nb)
                    seen.add(nb)
                if len(start_lanes) >= 12:
                    break

            def lane_length_xy(poly_xy: np.ndarray) -> float:
                return float(np.sum(np.linalg.norm(np.diff(poly_xy, axis=0), axis=1)))

            # DFS along exit_lanes topology
            all_paths: List[List[int]] = []

            # Build a quick route index for successor priority.
            route_index = {lid: i for i, lid in enumerate(route_seq)} if route_seq else {}

            def _ordered_successors(cur: int) -> List[int]:
                succs = [s for s in lane_graph.successors(cur) if s in lane_graph.lanes]
                if not succs:
                    return []
                if not route_seq or cur not in route_index:
                    return succs
                i = route_index[cur]
                preferred = route_seq[i + 1] if i + 1 < len(route_seq) else None
                if preferred is not None and preferred in succs:
                    # Put the route successor first, keep others as alternatives.
                    others = [s for s in succs if s != preferred]
                    return [preferred] + others
                return succs

            def dfs(path: List[int], acc_len: float, offroute_budget: int):
                cur = path[-1]
                if acc_len >= radius:
                    all_paths.append(path)
                    return
                succs = _ordered_successors(cur)
                if not succs:
                    all_paths.append(path)
                    return
                expanded = False
                for nxt in succs:
                    if nxt in path:
                        continue
                    # If route is available, strongly prefer staying on route; allow a very limited number
                    # of off-route alternatives (e.g., neighbor lane) to avoid empty expansions.
                    if route_seq and nxt not in route_set:
                        # allow neighbor lanes only
                        allow_offroute = (nxt in neighbor_extra) and (offroute_budget > 0)
                        if not allow_offroute:
                            continue
                    poly_xy = lane_graph.lanes[nxt].centerline
                    if poly_xy is None or len(poly_xy) < 2:
                        continue
                    # keep within radius by endpoint proximity
                    if float(np.min(np.linalg.norm(poly_xy - ego_pos[None, :2], axis=1))) > radius * 1.2:
                        continue
                    expanded = True
                    next_budget = offroute_budget
                    if route_seq and (nxt not in route_set) and (nxt not in neighbor_extra):
                        next_budget -= 1
                    dfs(path + [nxt], acc_len + lane_length_xy(poly_xy), next_budget)
                if not expanded:
                    all_paths.append(path)

            for lid in start_lanes:
                # if route_seq exists, we can require start lane in trimmed route.
                if route_seq and lid not in route_set:
                    continue
                poly_xy = lane_graph.lanes[lid].centerline
                if poly_xy is None or len(poly_xy) < 2:
                    continue
                dfs([lid], lane_length_xy(poly_xy), offroute_budget=1)

            # Merge lane centerlines into continuous polylines. Preserve original heading from polyline if present.
            reference_lines = []
            reference_line_route_ids = []

            def _dedup_append(lines: List[np.ndarray], ids: List[int], line: np.ndarray, route_id: int, eps: float = 1.0) -> None:
                """Append line if it's not covered by existing ones. Remove existing ones covered by this line."""
                if line is None or len(line) < 2:
                    return
                to_remove = []
                for i, ex in enumerate(lines):
                    diff = line[:, None, :2] - ex[None, :, :2]
                    dists = np.linalg.norm(diff, axis=-1)

                    dist_line_to_ex = np.max(np.min(dists, axis=1))
                    dist_ex_to_line = np.max(np.min(dists, axis=0))

                    if dist_line_to_ex < eps:
                        # 'line' is completely covered by 'ex' (within eps), so it provides no new path info
                        return

                    if dist_ex_to_line < eps:
                        # 'ex' is completely covered by 'line', mark 'ex' for replacement
                        to_remove.append(i)

                # Pop out in reverse to maintain indices
                for i in reversed(to_remove):
                    lines.pop(i)
                    ids.pop(i)

                lines.append(line)
                ids.append(route_id)

            for path in all_paths:
                merged = []
                for idx, lid in enumerate(path):
                    # We still have original polyline with heading in map_features_list.
                    raw = map_features_list.get(str(lid), None)
                    poly = None
                    if isinstance(raw, dict):
                        poly = raw.get('polyline', None)
                    if poly is None:
                        # fallback to xy-only centerline
                        poly_xy = lane_graph.lanes[lid].centerline
                        diff_y = np.diff(poly_xy[:, 1])
                        diff_x = np.diff(poly_xy[:, 0])
                        seg_hdg = np.arctan2(diff_y, diff_x)
                        heading = np.pad(seg_hdg, (1, 0), mode='edge')
                        poly = np.concatenate([poly_xy, heading[:, None]], axis=1)
                    poly = np.asarray(poly)
                    if idx > 0 and len(poly) > 1:
                        poly = poly[1:]
                    merged.append(poly)
                if not merged:
                    continue
                line = np.concatenate(merged, axis=0)
                _dedup_append(reference_lines, reference_line_route_ids, line, int(path[0]))

                # Cap number of reference lines to keep tensors bounded
                if len(reference_lines) >= 12:
                    break

            if len(reference_lines) == 0:
                # Last resort: allow unfiltered expansion (ignore route_set)
                for lid in start_lanes:
                    poly = map_features_list.get(str(lid), {}).get('polyline', None)
                    if poly is None:
                        continue
                    reference_lines.append(np.asarray(poly))
                    reference_line_route_ids.append(int(lid))

        # =========================
        # 6. 转成 feature tensor
        # =========================
        n_points = int(radius)
        M = len(reference_lines)

        position = np.zeros((M, n_points, 2))
        vector = np.zeros((M, n_points, 2))
        orientation = np.zeros((M, n_points))
        valid_mask = np.zeros((M, n_points), dtype=bool)

        future_projection = np.zeros((M, 8, 2))

        # route id per reference line
        route_id = np.full((M,), -1, dtype=np.int64)
        if 'reference_line_route_ids' in locals() and len(reference_line_route_ids) == M:
            route_id[:] = np.asarray(reference_line_route_ids, dtype=np.int64)

        future_samples = []
        ego_future = ego_features["position"][self.history_samples + 1:]
        if len(ego_future) > 0:
            future_samples = ego_future[9::10]  # 1Hz
            future_samples = [Point(xy) for xy in future_samples]

        for i, line in enumerate(reference_lines):
            line = np.asarray(line)
            subsample = line[::4][: n_points + 1]
            n_valid = len(subsample)

            position[i, : n_valid - 1] = subsample[:-1, :2]
            vector[i, : n_valid - 1] = np.diff(subsample[:, :2], axis=0)

            if line.shape[1] >= 3:
                orientation[i, : n_valid - 1] = subsample[:-1, 2]
            else:
                orientation[i, : n_valid - 1] = np.arctan2(
                    vector[i, : n_valid - 1, 1],
                    vector[i, : n_valid - 1, 0],
                )
            valid_mask[i, : n_valid - 1] = True

            # =========================
            # Pad the rest with the last valid point to prevent bent lines
            # =========================
            if n_valid - 1 > 0 and n_valid - 1 < n_points:
                position[i, n_valid - 1:] = position[i, n_valid - 2]
                vector[i, n_valid - 1:] = vector[i, n_valid - 2]
                orientation[i, n_valid - 1:] = orientation[i, n_valid - 2]

            if len(ego_future) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    linestring = LineString(line[:, :2])

                    for j, future_sample in enumerate(future_samples[:8]):
                        future_projection[i, j, 0] = linestring.project(future_sample)
                        future_projection[i, j, 1] = linestring.distance(future_sample)

        return {
            "position": position,
            "vector": vector,
            "orientation": orientation,
            "valid_mask": valid_mask,
            "future_projection": future_projection,
            # "route_id": route_id,
        }