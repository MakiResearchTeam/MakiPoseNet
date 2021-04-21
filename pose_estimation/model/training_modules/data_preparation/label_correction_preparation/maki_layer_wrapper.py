import tensorflow as tf


class MakiLayerWrapper:

    def __init__(self, data_tensor: tf.Tensor):
        self._data_tensor = data_tensor

    def get_data_tensor(self) -> tf.Tensor:
        return self._data_tensor
