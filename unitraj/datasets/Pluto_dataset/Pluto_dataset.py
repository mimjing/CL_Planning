import warnings

import traceback

import os
import pickle
import numpy as np
import h5py

from unitraj.datasets.Pluto_dataset.cost_map_manager import CostMapManager
from unitraj.datasets.Pluto_dataset.pluto_utils import _is_lane_like
from unitraj.datasets.Pluto_dataset.utils import save_dict_to_hdf5
from unitraj.datasets.base_dataset import BaseDataset
from scenarionet.common_utils import read_scenario

from shapely.geometry import Polygon
from shapely.geometry import Point, LineString
from typing import Optional, List, Tuple
from unitraj.datasets.Pluto_dataset.pluto_utils import parse_tracks_to_states, get_route_roadblock_ids, \
    calculate_additional_ego_states, PlutoFeature, interpolate_polyline




def _get_ego_features(state, ego_category_idx: int = 0, present_idx: int = 20, history_samples: int = 20, future_samples: int = 79):
    pos = state['position']
    
    # Calculate the valid slicing indices based on present_idx
    history_start = max(0, present_idx - history_samples)
    pad_front = max(0, history_samples - present_idx)
    future_end = present_idx + 1 + future_samples
    
    pos = pos[history_start:future_end]
    
    T = len(pos)
    position = pos[..., :2] if pos.shape[-1] >= 2 else pos
    heading = state['heading'][history_start:future_end]
    vel = state['velocity'][history_start:future_end]
    velocity = vel[..., :2] if vel.shape[-1] >= 2 else vel

    if 'acceleration' in state:
        accel = state['acceleration'][history_start:future_end]
        acceleration = accel[..., :2] if accel.shape[-1] >= 2 else accel
    else:
        acceleration = np.zeros((T, 2), dtype=np.float64)

    width = np.array(state['width'][history_start:future_end]).reshape(-1, 1)
    length = np.array(state['length'][history_start:future_end]).reshape(-1, 1)
    shape = np.concatenate([width, length], axis=-1)

    valid_mask = state['valid'][history_start:future_end]
    category = np.array(ego_category_idx, dtype=np.int8)

    # Pad the front if history_start was constrained by 0 (i.e. at the beginning of the scenario)
    if pad_front > 0:
        position = np.pad(position, ((pad_front, 0), (0, 0)), mode='edge')
        heading = np.pad(heading, (pad_front, 0), mode='edge')
        velocity = np.pad(velocity, ((pad_front, 0), (0, 0)), mode='constant')
        acceleration = np.pad(acceleration, ((pad_front, 0), (0, 0)), mode='constant')
        shape = np.pad(shape, ((pad_front, 0), (0, 0)), mode='edge')
        valid_mask = np.pad(valid_mask, (pad_front, 0), mode='constant', constant_values=False)

    return {
        "position": position.astype(np.float64),
        "heading": heading.astype(np.float64),
        "velocity": velocity.astype(np.float64),
        "acceleration": acceleration.astype(np.float64),
        "shape": shape.astype(np.float64),
        "category": category,
        "valid_mask": valid_mask,
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

        self.max_agents = 64
        self.max_static_obstacles = 10
        self.max_map_points = 3000
        self.max_polylines = 256
        self.history_length = 11  # Assuming standard
        self.num_points_polyline = 30
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

    def preprocess(self, scenario):
        """
        In UniTraj, preprocess often generates intermediate dict. 
        Here we skip intermediate formatting and construct pluto feature dict explicitly.
        """
        # We can either return the raw PlutoFeature or dict that HDF5 caches.
        # But HDF5 can't cache PlutoFeature directly if it contains custom objects.
        # Let's return primitive dict.
        self.ego_params = scenario['metadata']['ego_vehicle_parameters']
        all_tracks = scenario['tracks']

        ego_state_list = parse_tracks_to_states(all_tracks)
        map_features_list = scenario['map_features']
        traffic_light_status = scenario['dynamic_map_states']
        return [ego_state_list, map_features_list, traffic_light_status, all_tracks]

    def process(self, data, current_step=20):
        [ego_state_list, map_features_list, traffic_light_status, all_tracks] = data
        
        # Use current_step as the present index instead of hardcoded self.history_samples
        present_idx = current_step
        total_steps = self.history_samples + 1 + self.future_samples
        present_idx = min(present_idx, len(ego_state_list) - 1)
        
        present_ego_state = ego_state_list[present_idx]
        query_xy = present_ego_state['position'][:2]

        route_roadblocks_ids = get_route_roadblock_ids(ego_state_list, map_features_list)

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
            future_samples=self.future_samples
        )
        # Align ego features to expected horizon length.
        for k in ["position", "heading", "velocity", "acceleration", "shape", "valid_mask"]:
            if k in ego_features:
                ego_features[k] = _pad_or_truncate(ego_features[k], total_steps, axis=0, pad_value=0.0)
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
            route_roadblock_ids=route_roadblocks_ids,
            traffic_light_status=traffic_light_status,
            radius=self.radius,
            present_idx=present_idx,
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
            route_roadblock_ids=route_roadblocks_ids,
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
        total_steps = self.history_samples + 1 + self.future_samples

        # Find valid non-ego agents at present_idx
        present_agents = []
        for obj_id, track in all_tracks.items():
            if str(obj_id) == 'ego' or str(obj_id) == self.ego_params:  # rough check for ego
                continue
            state = track['state']
            if state['valid'][present_idx]:
                pos = state['position'][present_idx][:2]
                dist = np.linalg.norm(np.array(pos) - np.array(query_xy))
                present_agents.append((dist, obj_id, track))
                
        present_agents.sort(key=lambda x: x[0])
        present_agents = present_agents[:self.max_agents]
        
        N = len(present_agents)
        
        position = np.zeros((N, total_steps, 2), dtype=np.float64)
        heading = np.zeros((N, total_steps), dtype=np.float64)
        velocity = np.zeros((N, total_steps, 2), dtype=np.float64)
        shape = np.zeros((N, total_steps, 2), dtype=np.float64)
        category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, total_steps), dtype=np.bool_)
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
            pad_front = max(0, self.history_samples - present_idx)
            future_end = present_idx + 1 + self.future_samples

            pos_seq = np.asarray(track_state['position'], dtype=np.float64)[history_start:future_end, :2]
            vel_seq = np.asarray(track_state['velocity'], dtype=np.float64)[history_start:future_end, :2]
            hdg_seq = np.asarray(track_state['heading'], dtype=np.float64)[history_start:future_end]
            vld_seq = np.asarray(track_state['valid'], dtype=bool)[history_start:future_end]

            if pad_front > 0:
                pos_seq = np.pad(pos_seq, ((pad_front, 0), (0, 0)), mode='edge')
                vel_seq = np.pad(vel_seq, ((pad_front, 0), (0, 0)), mode='constant')
                hdg_seq = np.pad(hdg_seq, (pad_front, 0), mode='edge')
                vld_seq = np.pad(vld_seq, (pad_front, 0), mode='constant', constant_values=False)

            if len(pos_seq) > total_steps:
                pos_seq = pos_seq[:total_steps]
                vel_seq = vel_seq[:total_steps]
                hdg_seq = hdg_seq[:total_steps]
                vld_seq = vld_seq[:total_steps]
            elif len(pos_seq) < total_steps:
                pad_back = total_steps - len(pos_seq)
                pos_seq = np.pad(pos_seq, ((0, pad_back), (0, 0)), mode='edge')
                vel_seq = np.pad(vel_seq, ((0, pad_back), (0, 0)), mode='constant')
                hdg_seq = np.pad(hdg_seq, (0, pad_back), mode='edge')
                vld_seq = np.pad(vld_seq, (0, pad_back), mode='constant', constant_values=False)

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
            try:
                # the present_idx in the sliced array is always self.history_samples
                if bool(vld_seq[self.history_samples]):
                    center_now = pos_seq[self.history_samples]
                    heading_now = float(hdg_seq[self.history_samples])
                    poly_xy = _box_corners_xy(center_now, heading_now, float(w0), float(l0))
                    polygon[idx] = Polygon(poly_xy)
                else:
                    polygon[idx] = None
            except Exception:
                polygon[idx] = None
            
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
        dynamic_types = {'VEHICLE', 'PEDESTRIAN', 'BICYCLE'}

        for obj_id, track in all_tracks.items():
            track_type_raw = track.get('type', '')
            track_type = str(track_type_raw).upper()
            if track_type in dynamic_types or track_type == 'EGO':
                continue

            state = track['state']
            if not state['valid'][present_idx]:
                continue

            pos = state['position'][present_idx][:2]
            dist = np.linalg.norm(np.array(pos) - np.array(query_xy))
            if dist > self.radius:
                continue

            heading = state['heading'][present_idx]
            length = state['length'][present_idx] if isinstance(state['length'], (list, np.ndarray)) else state[
                'length']
            width = state['width'][present_idx] if isinstance(state['width'], (list, np.ndarray)) else state[
                'width']

            # Map static obstacle type to pluto_feature_builder indices
            if "TRAFFIC_BARRIER" in track_type:
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
    ):
        present_idx_use = present_idx if present_idx is not None else self.history_samples
        route_ids = set(str(route_id) for route_id in route_roadblock_ids)

        # traffic_light_status (SD) is typically a dict keyed by lane_connector_id/lane_id.
        # We convert it to lane_id -> numeric status at present_idx.
        # Encoding (align with other UniTraj datasets):
        #   0: UNKNOWN, 1: GREEN, 2: YELLOW, 3: RED
        state_mapping = {
            "TRAFFIC_LIGHT_UNKNOWN": 0,
            "UNKNOWN": 0,
            "0": 0,
            "TRAFFIC_LIGHT_GREEN": 1,
            "GREEN": 1,
            "1": 1,
            "TRAFFIC_LIGHT_YELLOW": 2,
            "YELLOW": 2,
            "2": 2,
            "TRAFFIC_LIGHT_RED": 3,
            "RED": 3,
            "3": 3,
        }

        def _tl_to_int(x) -> int:
            if x is None:
                return 0
            if isinstance(x, dict):
                # sometimes {'status': 'TRAFFIC_LIGHT_RED'}
                x = x.get('status', x.get('state', x))
            if isinstance(x, (np.integer, int)):
                # If another exporter already uses ints, clamp to [0..3]
                v = int(x)
                return v if 0 <= v <= 3 else 0
            s = str(x).upper()
            return int(state_mapping.get(s, 0))

        tls = {}
        for lane_id, tl_info in (traffic_light_status or {}).items():
            if not isinstance(tl_info, dict):
                tls[str(lane_id)] = _tl_to_int(tl_info)
                continue
            state = tl_info.get('state', {}) if isinstance(tl_info.get('state', None), dict) else {}
            obj_state = state.get('object_state', None)
            if obj_state is None:
                obj_state = tl_info.get('object_state', None)

            if isinstance(obj_state, (list, tuple, np.ndarray)) and len(obj_state) > present_idx_use:
                tls[str(lane_id)] = _tl_to_int(obj_state[present_idx_use])
            else:
                tls[str(lane_id)] = _tl_to_int(obj_state)

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
            else:
                lane_objects.append((map_id, map_feat))

        object_ids = [map_id for map_id, _ in lane_objects + crosswalk_objects]

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
            # Since we lack explicit left/right boundaries, we just duplicate centerline for now
            left_bound = centerline.copy()
            right_bound = centerline.copy()
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
            rec_polygon_type.append(int(self.polygon_types.index("LANE")))
            rec_polygon_on_route.append(bool(str(map_id) in route_ids))
            rec_polygon_tl_status.append(int(tls.get(str(map_id), 0)))
            rec_polygon_has_speed_limit.append(False)
            rec_polygon_speed_limit.append(0.0)
            rec_polygon_road_block_id.append(int(hash(str(map_id)) % (10**8)))

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
            rec_polygon_road_block_id.append(int(hash(str(map_id)) % (10**8)))

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
                d = np.linalg.norm(rel, axis=1)
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

    # def _get_reference_line_feature(
    #     self,
    #     ego_features: dict,
    #     map_features_list: dict,
    #     route_roadblock_ids=None,
    #     radius: Optional[float] = None,
    #     n_points: Optional[int] = None,
    #     max_reference_lines: int = 12,
    #     present_idx: Optional[int] = None,
    # ):
    #     """Build reference line candidates from SD map_features.
    #
    #     Output matches Pluto planning head expectations:
    #       position: (R, P, 2)
    #       vector: (R, P, 2)
    #       orientation: (R, P)
    #       valid_mask: (R, P)
    #       future_projection: (R, 8, 2)  # [s_on_line(m), dist_to_line(m)]
    #
    #     Notes:
    #       * No nuplan ScenarioManager is required.
    #       * We use lane/lane_connector polylines as candidate reference lines.
    #       * If route_roadblock_ids is provided, we prefer lanes on the route.
    #     """
    #
    #     radius = float(radius if radius is not None else self.radius)
    #     # Pluto builder uses 1m spacing: n_points = int(radius / 1.0)
    #     n_points = int(n_points if n_points is not None else max(int(radius / 1.0), 1))
    #
    #     present_idx_use = present_idx if present_idx is not None else self.history_samples
    #     ego_xy = np.asarray(ego_features["position"][present_idx_use], dtype=np.float64)[:2]
    #     ego_heading = float(np.asarray(ego_features["heading"][present_idx_use]))
    #
    #     def _polyline_to_xyh(polyline: np.ndarray) -> np.ndarray:
    #         arr = np.asarray(polyline)
    #         if arr.ndim != 2 or arr.shape[0] < 2:
    #             return np.zeros((0, 3), dtype=np.float64)
    #         xy = arr[:, :2].astype(np.float64)
    #         dxy = np.diff(xy, axis=0)
    #         yaw = np.arctan2(dxy[:, 1], dxy[:, 0])
    #         yaw = np.concatenate([yaw, yaw[-1:]], axis=0)
    #         return np.concatenate([xy, yaw[:, None]], axis=-1)
    #
    #     # 1) collect candidate polylines
    #     candidates: List[Tuple[float, np.ndarray]] = []  # (score, line_xyh)
    #     on_route: List[Tuple[float, np.ndarray]] = []
    #
    #     route_set = set(int(x) for x in route_roadblock_ids) if route_roadblock_ids else None
    #
    #     for map_id, feat in (map_features_list or {}).items():
    #         mtype = str(feat.get("type", "")).upper()
    #         if not _is_lane_like(mtype): continue
    #
    #         line_xyh = _polyline_to_xyh(feat["polyline"])
    #         if line_xyh.shape[0] < 2: continue
    #
    #         # distance score: min distance from ego to polyline vertices
    #         d = np.linalg.norm(line_xyh[:, :2] - ego_xy[None, :], axis=-1).min()
    #         if d > radius: continue
    #
    #         score = float(d)
    #
    #         is_on_route = False
    #         if route_set is not None:
    #             if int(map_id) in route_set: is_on_route = True
    #
    #         if is_on_route:
    #             on_route.append((score, line_xyh))
    #         else:
    #             candidates.append((score, line_xyh))
    #
    #     # prefer on-route, then nearest
    #     selected = sorted(on_route, key=lambda x: x[0]) + sorted(candidates, key=lambda x: x[0])
    #     selected = selected[:max_reference_lines]
    #
    #     # 2) fallback: straight line along ego heading
    #     if len(selected) == 0:
    #         xs = ego_xy[0] + np.linspace(0.0, radius, n_points + 1) * np.cos(ego_heading)
    #         ys = ego_xy[1] + np.linspace(0.0, radius, n_points + 1) * np.sin(ego_heading)
    #         line_xyh = np.stack([xs, ys, np.full_like(xs, ego_heading)], axis=-1)
    #         selected = [(0.0, line_xyh)]
    #
    #     R = len(selected)
    #     P = n_points
    #
    #     position = np.zeros((R, P, 2), dtype=np.float64)
    #     vector = np.zeros((R, P, 2), dtype=np.float64)
    #     orientation = np.zeros((R, P), dtype=np.float64)
    #     valid_mask = np.zeros((R, P), dtype=np.bool_)
    #     future_projection = np.zeros((R, 8, 2), dtype=np.float64)
    #
    #     ego_future = np.asarray(ego_features["position"][present_idx + 1 :], dtype=np.float64)
    #     if ego_future.ndim == 2 and ego_future.shape[-1] >= 2 and ego_future.shape[0] > 0:
    #         # every 1s (dt=0.1 => step 10)
    #         future_samples_xy = ego_future[9::10, :2]
    #     else:
    #         future_samples_xy = np.zeros((0, 2), dtype=np.float64)
    #
    #     for i, (_score, line_xyh) in enumerate(selected):
    #         # resample to (P+1) points using arc-length interpolation
    #         # line_xyh: (N,3). We interpolate on xy, then recompute yaw.
    #         xy = line_xyh[:, :2]
    #         if xy.shape[0] < 2: continue
    #
    #         xy_rs = interpolate_polyline(xy, P + 1).astype(np.float64)  # (P+1,2)
    #         dxy = np.diff(xy_rs, axis=0)
    #         yaw = np.arctan2(dxy[:, 1], dxy[:, 0])
    #
    #         position[i] = xy_rs[:-1]
    #         vector[i] = dxy
    #         orientation[i] = yaw
    #         valid_mask[i] = True
    #
    #         if future_samples_xy.shape[0] > 0:
    #             # Use shapely for (s, dist) projection like builder
    #             ls = LineString(xy_rs)
    #             for j in range(min(8, future_samples_xy.shape[0])):
    #                 pt = Point(float(future_samples_xy[j, 0]), float(future_samples_xy[j, 1]))
    #                 try:
    #                     future_projection[i, j, 0] = float(ls.project(pt))
    #                     future_projection[i, j, 1] = float(ls.distance(pt))
    #                 except Exception:
    #                     # leave zeros on any numerical issues
    #                     pass
    #     return {
    #         "position": position,
    #         "vector": vector,
    #         "orientation": orientation,
    #         "valid_mask": valid_mask,
    #         "future_projection": future_projection,
    #     }


    def _get_reference_line_feature(
        self,
        ego_features,
        map_features_list,
        route_roadblock_ids=None,
        training=True,
    ):
        ego_pos = ego_features["position"][self.history_samples]
        ego_heading = ego_features["heading"][self.history_samples]

        radius = self.radius

        # route_roadblock_ids is the whole-scenario route id sequence (provided by get_route_roadblock_ids).
        # We trim to only keep ids from ego current position onward.
        route_roadblock_ids = route_roadblock_ids or []
        route_seq = [int(x) for x in route_roadblock_ids]

        # 1. 提取 lane segments
        lanes = []
        for map_id, feat in map_features_list.items():
            t = str(feat.get("type", "")).upper()
            if "LANE" not in t:
                continue
            poly = feat.get("polyline", None)
            if poly is None or len(poly) < 2:
                continue
            poly = np.asarray(poly)

            # Limit lane extraction to within `radius` of ego to avoid excessive computation.
            # Use minimum distance to polyline points as an inexpensive filter.
            try:
                min_dist = np.min(np.linalg.norm(poly[:, :2] - ego_pos[None, :2], axis=1))
            except Exception:
                continue
            if min_dist > radius:
                continue
            start = poly[0, :2]
            end = poly[-1, :2]

            def compute_heading(p1, p2):
                vec = p2 - p1
                return np.arctan2(vec[1], vec[0])
            heading_start = compute_heading(poly[0, :2], poly[1, :2])
            heading_end = compute_heading(poly[-2, :2], poly[-1, :2])

            lanes.append({
                "poly": poly,
                "start": start,
                "end": end,
                "heading_start": heading_start,
                "heading_end": heading_end,
                # treat map_id as route_id/roadblock_id (same assumption as get_route_roadblock_ids)
                "route_id": int(map_id) if str(map_id).isdigit() else (hash(str(map_id)) % (10**8)),
            })

        # =========================
        # 2. 构建伪 graph（lane stitching）
        # =========================
        def wrap_to_pi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi
        adj = {i: [] for i in range(len(lanes))}

        for i, A in enumerate(lanes):
            for j, B in enumerate(lanes):
                if i == j:
                    continue

                dist = np.linalg.norm(A["end"] - B["start"])
                heading_diff = abs(wrap_to_pi(A["heading_end"] - B["heading_start"]))

                if dist < 3.0 and heading_diff < 0.52:  # ≈30°
                    adj[i].append(j)

        # =========================
        # 3. 找 ego 所在 lane（过滤对向）
        # =========================
        def point_to_polyline_distance(point, polyline):
            return np.min(np.linalg.norm(polyline[:, :2] - point[None, :], axis=1))

        # candidates = []
        # ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)])
        # for i, lane in enumerate(lanes):
        #     poly = lane["poly"]
        #     dist = point_to_polyline_distance(ego_pos, poly)
        #     heading_diff = abs(wrap_to_pi(lane["heading_start"] - ego_heading))
        #     # forward check
        #     vec = poly[0, :2] - ego_pos
        #     forward = np.dot(vec, ego_dir)
        #     if dist < 5.0 and heading_diff < 0.78 and forward > -2.0:  # 45°
        #         candidates.append((i, dist))

        candidates = []
        ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)
        # 可调阈值：先放宽一点，避免 candidates 为空导致 fallback
        DIST_TH = 8.0  # 原来 5.0，点集距离不准时容易误杀
        HEADING_TH = 1.05  # 约 60°，路口/弯道更稳；原来 45°(0.78)
        FORWARD_TH = -2.0  # 仍保留“允许在身后 2m 内”的宽容
        for i, lane in enumerate(lanes):
            poly = np.asarray(lane["poly"])
            if poly is None or len(poly) < 2:
                continue
            # 1) 用“点到点集最小距离”作为粗dist（保留你原来的轻量实现）
            #    如果你愿意更准，可用 shapely LineString(poly[:,:2]).distance(Point(...))
            dists = np.linalg.norm(poly[:, :2] - ego_pos[None, :2], axis=1)
            k = int(np.argmin(dists))  # poly 上离 ego 最近的点索引
            dist = float(dists[k])

            if dist > DIST_TH:
                continue
            # 2) 用最近点附近的局部切向估计 lane 朝向（避免用 poly[0] 造成误杀）
            #    取 k 的前后点做差，避免 k 在边界时越界
            if k == 0:
                p0 = poly[0, :2]
                p1 = poly[1, :2]
            elif k >= len(poly) - 1:
                p0 = poly[-2, :2]
                p1 = poly[-1, :2]
            else:
                p0 = poly[k - 1, :2]
                p1 = poly[k + 1, :2]
            traj_vec = (p1 - p0).astype(np.float64)
            traj_norm = float(np.linalg.norm(traj_vec))
            if traj_norm < 1e-3:
                # 局部退化，跳过
                continue
            traj_dir = traj_vec / traj_norm
            traj_heading = float(np.arctan2(traj_dir[1], traj_dir[0]))
            heading_diff = abs(wrap_to_pi(traj_heading - ego_heading))
            if heading_diff > HEADING_TH:
                continue
            # 3) forward 判定也用“最近点”而不是 poly[0]
            vec_near = (poly[k, :2] - ego_pos[:2]).astype(np.float64)
            forward = float(np.dot(vec_near, ego_dir))
            if forward <= FORWARD_TH:
                continue
            candidates.append((i, dist))

        # Determine ego current route_id from the lane candidates that ego lies on.
        # Then trim the whole-scenario route_seq to start from that route_id.
        if route_seq:
            cur_route_candidates = [lanes[i]["route_id"] for i, _ in candidates] if 'candidates' in locals() else []
            start_idx = None
            for rid in cur_route_candidates:
                if rid in route_seq:
                    start_idx = route_seq.index(rid)
                    break
            if start_idx is not None:
                route_seq = route_seq[start_idx:]

        # fallback：如果没有候选
        if len(candidates) == 0:
            print('—————————————— Warning,参考线为直线————————————')
            xs = ego_pos[0] + np.linspace(0, radius, int(radius)) * np.cos(ego_heading)
            ys = ego_pos[1] + np.linspace(0, radius, int(radius)) * np.sin(ego_heading)
            fake_line = np.stack([xs, ys, np.full_like(xs, ego_heading)], axis=-1)
            reference_lines = [fake_line]
            reference_line_route_ids = [route_seq[0] if route_seq else -1]
        else:
            # 选最近的几个
            candidates = sorted(candidates, key=lambda x: x[1])[:3]
            start_indices = [c[0] for c in candidates]

            # =========================
            # 4. DFS rollout paths
            # =========================
            def lane_length(idx):
                poly = lanes[idx]["poly"]
                return np.sum(np.linalg.norm(np.diff(poly[:, :2], axis=0), axis=1))

            def expand_path(start_idx):
                paths = []

                def dfs(path, length):
                    cur = path[-1]

                    if length > radius:
                        paths.append(path)
                        return

                    if len(adj[cur]) == 0:
                        paths.append(path)
                        return

                    for nxt in adj[cur]:
                        if nxt in path:  # avoid loop
                            continue
                        dfs(path + [nxt], length + lane_length(nxt))

                dfs([start_idx], 0)
                return paths

            all_paths = []
            for s in start_indices:
                all_paths.extend(expand_path(s))

            # =========================
            # 5. merge polyline
            # =========================
            def merge_path(path):
                merged = []
                for idx in path:
                    poly = lanes[idx]["poly"]
                    if len(merged) > 0:
                        poly = poly[1:]
                    merged.append(poly)
                return np.concatenate(merged, axis=0)

            # Filter out paths that start with a route_id behind ego (not in trimmed route_seq).
            reference_lines = []
            reference_line_route_ids = []
            route_set = set(route_seq)
            for p in all_paths:
                start_rid = lanes[p[0]]["route_id"]
                if route_seq and start_rid not in route_set:
                    continue
                reference_lines.append(merge_path(p))
                reference_line_route_ids.append(start_rid)

            # If everything got filtered, fall back to unfiltered paths (but still output route ids).
            if len(reference_lines) == 0:
                reference_lines = [merge_path(p) for p in all_paths]
                reference_line_route_ids = [lanes[p[0]]["route_id"] for p in all_paths]

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

        # future（仅训练用）
        if training:
            ego_future = ego_features["position"][self.history_samples + 1:]
            if len(ego_future) > 0:
                future_samples = ego_future[9::10]  # 1Hz
                future_samples = [Point(xy) for xy in future_samples]
        else:
            ego_future = []

        for i, line in enumerate(reference_lines):
            line = np.asarray(line)

            # subsample
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

            # future projection（训练用）
            if training and len(ego_future) > 0:
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
            "route_id": route_id,
        }