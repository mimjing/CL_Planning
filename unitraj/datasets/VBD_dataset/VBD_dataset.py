import fcntl
import os, pickle, h5py
import numpy as np
import torch

from unitraj.datasets.base_dataset import BaseDataset
from unitraj.datasets.VBD_dataset.VBDdata_utils import data_process_scenario
from scenarionet.common_utils import read_scenario


class VBDDataset(BaseDataset):
    """
    直接输入 Scenario 原始 pkl，内部转 HDF5 缓存，其余逻辑 100% 复用 BaseDataset
    """
    def __init__(self, config, is_validation=False):
        # 1. 让父类跳过“读 HDF5”分支，落到我们重写的 _get_scenario
        config['use_cache'] = False          # 强制走原始数据分支
        config['overwrite_cache'] = True     # 每次启动自动写缓存
        self.anchors = pickle.load(open(config['anchor_path'], "rb"))
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
                    internal = self.process(scenario)
                    output = [self.post_process(internal)]

                except Exception as e:
                    print('Warning: {} in {}'.format(e, file_name))
                    output = None

                if output is None: continue

                for i, record in enumerate(output):
                    grp_name = dataset_name + '-' + str(worker_index) + '-' + str(cnt) + '-' + str(i)
                    grp = f.create_group(grp_name)
                    for key, value in record.items():
                        if isinstance(value, str):
                            value = np.bytes_(value)
                        grp.create_dataset(key, data=value)
                    file_info = {}
                    file_info['h5_path'] = hdf5_path
                    file_list[grp_name] = file_info
                del scenario
                del output

        return file_list

    def _get_scenario(self, idx):
        """返回 List[dict]（与 BaseDataset.process 输出一致）"""
        pkl_path = self.pkl_lookup[idx]
        h5_path  = self.data_loaded[self.data_loaded_keys[idx]]['h5_path']

        # 1. 缓存命中：直接读 h5（复用父类逻辑）
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                records = []
                for gid in sorted(f.keys(), key=int):
                    grp = f[gid]
                    rec = {k: (grp[k][()].decode('utf-8') if grp[k].dtype.type == np.bytes_ else grp[k][()])
                           for k in grp.keys()}
                    records.append(rec)
                return records

        # 2. 缓存未命中：读 pkl → 处理 → 写 h5 → 返回记录
        return self._build_and_cache(h5_path, pkl_path)

    # ------------------------------------------------------------------
    # 真正处理 + 缓存
    # ------------------------------------------------------------------
    def _build_and_cache(self, h5_path, pkl_path):
        """线程/进程安全：先写临时文件，再原子 rename"""
        tmp_path = h5_path + '.tmp'

        # 文件锁（可选）：保证多进程不重复写同一份
        lock_path = h5_path + '.lock'
        with open(lock_path, 'w') as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                if os.path.exists(h5_path):  # 二次检查
                    return self._get_scenario(self.pkl_lookup.index(pkl_path))
                self._write_h5(tmp_path, pkl_path)
                os.rename(tmp_path, h5_path)  # 原子操作
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                try:
                    os.remove(lock_path)
                except:
                    pass

        return self._get_scenario(self.pkl_lookup.index(pkl_path))

    def _write_h5(self, h5_path, pkl_path):
        """与 BaseDataset 处理流程完全一致"""
        with open(pkl_path, 'rb') as f:
            scenario = pickle.load(f)

        # 三段式
        base = BaseDataset.__new__(BaseDataset)
        internal = base.preprocess(scenario)
        records = base.process(internal)
        records = base.postprocess(records)

        with h5py.File(h5_path, 'w') as f:
            for i, rec in enumerate(records):
                grp = f.create_group(f'{i:04d}')
                for k, v in rec.items():
                    grp.create_dataset(k, data=np.bytes_(v) if isinstance(v, str) else v)

    def process(self, scenario):
        data_dict = data_process_scenario(
            scenario,
            max_num_objects=16,
            max_polylines=256,
            current_index=10,
            num_points_polyline=30,
        )
        return data_dict

    def _process(self, types):
        """
        Process the agent types and convert them into anchor vectors.

        Args:
            types (numpy.ndarray): Array of agent types.

        Returns:
            numpy.ndarray: Array of anchor vectors.
        """
        anchors = []

        for i in range(len(types)):
            if types[i] == 0:
                anchors.append(self.anchors['TYPE_VEHICLE'])
            elif types[i] == 1:
                anchors.append(self.anchors['TYPE_PEDESTRIAN'])
            elif types[i] == 2:
                anchors.append(self.anchors['TYPE_CYCLIST'])
            else:
                anchors.append(np.zeros_like(self.anchors['TYPE_VEHICLE']))  # 能否增加trafficcone和trafficbarrier的类型呢

        return np.array(anchors, dtype=np.float32)

    def post_process(self, data):
        """
        Generate tensors from the input data.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Dictionary of tensors.
        """

        agents_history = data['agents_history']
        agents_interested = data['agents_interested']
        agents_future = data['agents_future']
        agents_type = data['agents_type']
        traffic_light_points = data['traffic_light_points']
        polylines = data['polylines']
        polylines_valid = data['polylines_valid']
        relations = data['relations']
        anchors = self._process(agents_type)

        tensors = {
            "agents_history": torch.from_numpy(agents_history),
            "agents_interested": torch.from_numpy(agents_interested),
            "agents_future": torch.from_numpy(agents_future),
            "agents_type": torch.from_numpy(agents_type),
            "traffic_light_points": torch.from_numpy(traffic_light_points),
            "polylines": torch.from_numpy(polylines),
            "polylines_valid": torch.from_numpy(polylines_valid),
            "relations": torch.from_numpy(relations),
            "anchors": torch.from_numpy(anchors)
        }

        return tensors

    def collate_fn(self, batch_list):
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
