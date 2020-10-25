from abc import ABC, abstractmethod
from makiflow.core import MakiModel


class PoseEstimatorInterface(MakiModel, ABC):
    @abstractmethod
    def get_paf_makitensors(self):
        pass

    @abstractmethod
    def get_heatmap_makitensors(self):
        pass
