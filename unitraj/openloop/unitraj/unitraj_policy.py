from unitraj.closeloop.unitraj.unitraj_inference import UnitrajInference
from unitraj.openloop.inference_policy import OpenInferencePolicy


class OpenUniTrajPolicy(OpenInferencePolicy):
    """
    继承InferencePolicy，实现UniTraj推理
    """

    def __init__(self, obj, seed):
        super(OpenUniTrajPolicy, self).__init__(obj, seed)

        self.Sim = UnitrajInference(self.cfg)
        self.Sim.initialize_model()
