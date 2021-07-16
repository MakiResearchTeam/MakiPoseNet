import tensorflow as tf

import makiflow as mf


class PeakExtraction(mf.core.MakiLayer):
    _DEFAULT_KERNEL_MAX_POOL = [1, 3, 3, 1]

    def __init__(self, threshold=0.1, name='PeakExtraction'):
        """

        Parameters
        ----------
        threshold : float
            Threshold if keypoint have bigger probability, it will be used, otherwise it will as not detected keypoint.
        name : str
            Layer name.
        """
        self._threshold_peaks = threshold
        super(PeakExtraction, self).__init__(name=name, params=[], regularize_params=[], named_params_dict={})

    def forward(self, x, computation_mode=mf.core.MakiLayer.INFERENCE_MODE):
        max_pooled_heatmap = tf.nn.max_pool(
            x,
            PeakExtraction._DEFAULT_KERNEL_MAX_POOL,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        # Take only values that equal to heatmap from max pooling,
        # i.e. biggest numbers of heatmaps
        peaks = tf.where(
            tf.equal(
                x,
                max_pooled_heatmap
            ),
            x,
            tf.zeros_like(x)
        )
        return peaks

    def training_forward(self, x):
        self.forward(x)

    @staticmethod
    def build(params: dict):
        pass

    def to_dict(self):
        pass
