import tensorflow as tf

import makiflow as mf


class IndicesScoreExtraction(mf.core.MakiLayer):
    def __init__(self, threshold=0.1, name='IndicesScoreExtraction'):
        """
        Used for inference only (it is assumed that the batchsize is 1).

        Parameters
        ----------
        threshold : float
            Threshold if keypoint have bigger probability, it will be used, otherwise it will as not detected keypoint.
        name : str
            Layer name.
        """
        self._threshold_peaks = threshold
        super(IndicesScoreExtraction, self).__init__(name=name, params=[], regularize_params=[], named_params_dict={})

    def forward(self, x, computation_mode=mf.core.MakiLayer.INFERENCE_MODE):
        flat_peaks = tf.reshape(x, [-1])
        coords = tf.range(0, tf.shape(flat_peaks)[0], dtype=tf.int32)

        peaks_coords = coords[flat_peaks > self._threshold_peaks]

        peaks = tf.gather(flat_peaks, peaks_coords)

        indices = tf.transpose(tf.unravel_index(peaks_coords, dims=tf.shape(x)), [1, 0])
        return indices, peaks

    def training_forward(self, x):
        return self.forward(x)

    @staticmethod
    def build(params: dict):
        pass

    def to_dict(self):
        pass
