from unitraj.closeloop.inference_policy import InferencePolicy
from unitraj.closeloop.VBD.vbd_inference import VBDInference

class VBDPolicy(InferencePolicy):
    """
    继承InferencePolicy，实现VBD推理
    """

    def __init__(self, obj, seed):
        super(VBDPolicy, self).__init__(obj, seed)

        self.Sim = VBDInference(self.cfg)
        self.Sim.initialize_model()
