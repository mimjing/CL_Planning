import time
import torch
import numpy as np
from torch.utils.data import Dataset

from unitraj.datasets.Pluto_dataset.Pluto_dataset import PlutoDataset
from unitraj.datasets.Pluto_dataset.pluto_utils import PlutoFeature
from unitraj.datasets.unitraj_test_dataset import create_batch_dict


class PlutoTestDataset(Dataset):
    """
    Dataset class for Pluto feature format reading NuPlan db files (ScenarioDescription).
    """
    def __init__(self, config=None, is_validation=False):
        if config is None:
            config = {}
        self._builder = PlutoDataset(config=config, is_validation=is_validation)
    
    def process_scenario(self, scenario, current_step):
        t1 = time.time()
        output = self.process_scenario_data(scenario, current_step)
        t2 = time.time()

        # PlutoDataset.postprocess returns a list (len=1) of sample dict(s)
        # ret_list = output if isinstance(output, list) else [output]
        # batch_dict = create_batch_dict(ret_list)

        return output

    def process_scenario_data(self, scenario, current_step):
        intermediate = self._builder.preprocess(scenario)
        pluto_feature = self._builder.process(intermediate, current_step=current_step)
        output = self._builder.postprocess(pluto_feature)
        return output

    
