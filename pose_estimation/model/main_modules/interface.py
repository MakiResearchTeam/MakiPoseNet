from abc import ABC, abstractmethod
from makiflow.base.maki_entities.maki_core import MakiCore


class PoseEstimatorInterface(MakiCore, ABC):

    @abstractmethod
    def training_on(self):
        pass

    @abstractmethod
    def predict(self, x: list):
        pass

    @abstractmethod
    def get_paf_tensors(self) -> list:
        pass

    @abstractmethod
    def get_heatmap_tensors(self) -> list:
        pass

    @abstractmethod
    def get_training_vars(self):
        pass

    @abstractmethod
    def build_final_loss(self, loss):
        pass

    @abstractmethod
    def get_session(self):
        # TODO: Move this method into MakiModel
        pass
