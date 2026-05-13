from unitraj.closeloop.inference_policy import InferencePolicy
from unitraj.closeloop.pluto.pluto_inference import PlutoInference
class PlutoPolicy(InferencePolicy):
    """
    Inherits InferencePolicy to implement Pluto inference
    """
    def __init__(self, obj, seed):
        super(PlutoPolicy, self).__init__(obj, seed)
        self.Sim = PlutoInference(self.ego_cfg)
        self.Sim.initialize_model()
