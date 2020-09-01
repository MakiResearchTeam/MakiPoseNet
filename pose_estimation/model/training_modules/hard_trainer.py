import tensorflow as tf
from ..main_modules import PoseEstimatorInterface


class HardTrainer:
    def __init__(self, model: PoseEstimatorInterface):
        self.__model = model

    def __build_loss(self):
        pass
