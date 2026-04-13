from unitraj.closeloop.VBD.vbd_inference import VBDInference
from unitraj.openloop.inference_policy import OpenInferencePolicy


class OpenVBDPolicy(OpenInferencePolicy):
    """
    继承InferencePolicy，实现VBD推理
    """

    def __init__(self, obj, seed):
        super(OpenVBDPolicy, self).__init__(obj, seed)

        self.Sim = VBDInference(self.cfg)
        self.Sim.initialize_model()
