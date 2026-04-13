from .MTR_dataset import MTRDataset
from .autobot_dataset import AutoBotDataset
from .VBD_dataset import VBDDataset
from .wayformer_dataset import WayformerDataset
from .SMART_dataset import SMARTDataset

__all__ = {
    'autobot': AutoBotDataset,
    'unitraj': WayformerDataset,
    'MTR': MTRDataset,
    'SMART': SMARTDataset,
    'VBD': VBDDataset,
}


def build_dataset(config, val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
