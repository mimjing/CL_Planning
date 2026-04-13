from unitraj.models.autobot.autobot import AutoBotEgo
from unitraj.models.mtr.MTR import MotionTransformer
from unitraj.models.vbd.model.vbd import VBD
from unitraj.models.wayformer.wayformer import Wayformer
from unitraj.models.smart.smart import SMART

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
    'SMART': SMART,
    'VBD': VBD,
}


def build_model(config):
    model = __all__[config.model_name](
        config=config
    )

    return model
