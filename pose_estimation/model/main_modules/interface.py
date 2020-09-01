from abc import ABC, abstractmethod
from makiflow.base.maki_entities.maki_core import MakiCore


class PoseEstimatorInterface(MakiCore, ABC):
    @abstractmethod
    def get_paf_tensor(self):
        pass

    @abstractmethod
    def get_heatmap_tensor(self):
        pass

    @abstractmethod
    def get_training_vars(self):
        pass

    @abstractmethod
    def build_final_loss(self, loss):
        pass
