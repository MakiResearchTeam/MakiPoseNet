import tensorflow as tf

import makiflow as mf


class PeakExtractionFast(mf.core.MakiLayer):
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
        super(PeakExtractionFast, self).__init__(name=name, params=[], regularize_params=[], named_params_dict={})

    def forward(self, x, computation_mode=mf.core.MakiLayer.INFERENCE_MODE):
        peaks = tf.vectorized_map(fn=self.generate_peaks, elems=x)
        return peaks

    def generate_peaks(self, heatmap):
        heatmap_tf = heatmap
        heatmap_tf = tf.where(
            tf.less(heatmap_tf, self._threshold_peaks),
            tf.zeros_like(heatmap_tf), heatmap_tf
        )

        heatmap_with_borders = tf.pad(heatmap_tf, [(2, 2), (2, 2), (0, 0)])
        heatmap_center = heatmap_with_borders[
                         1:tf.shape(heatmap_with_borders)[0] - 1,
                         1:tf.shape(heatmap_with_borders)[1] - 1
                         ]
        heatmap_left = heatmap_with_borders[
                       1:tf.shape(heatmap_with_borders)[0] - 1,
                       2:tf.shape(heatmap_with_borders)[1]
                       ]
        heatmap_right = heatmap_with_borders[
                        1:tf.shape(heatmap_with_borders)[0] - 1,
                        0:tf.shape(heatmap_with_borders)[1] - 2
                        ]
        heatmap_up = heatmap_with_borders[
                     2:tf.shape(heatmap_with_borders)[0],
                     1:tf.shape(heatmap_with_borders)[1] - 1
                     ]
        heatmap_down = heatmap_with_borders[
                       0:tf.shape(heatmap_with_borders)[0] - 2,
                       1:tf.shape(heatmap_with_borders)[1] - 1
                       ]

        peaks = (heatmap_center > heatmap_left) & \
                (heatmap_center > heatmap_right) & \
                (heatmap_center > heatmap_up) & \
                (heatmap_center > heatmap_down)
        return tf.cast(peaks, tf.float32)

    def training_forward(self, x):
        self.forward(x)

    @staticmethod
    def build(params: dict):
        pass

    def to_dict(self):
        pass
