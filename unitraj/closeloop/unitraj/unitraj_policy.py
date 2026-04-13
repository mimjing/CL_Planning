from unitraj.closeloop.inference_policy import InferencePolicy
from unitraj.closeloop.unitraj.unitraj_inference import UnitrajInference

class UniTrajPolicy(InferencePolicy):
    """
    继承InferencePolicy，实现UniTraj推理
    """

    def __init__(self, obj, seed):
        super(UniTrajPolicy, self).__init__(obj, seed)

        self.Sim = UnitrajInference(self.cfg)
        self.Sim.initialize_model()
