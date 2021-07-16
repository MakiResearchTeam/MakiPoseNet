import tensorflow as tf

import makiflow as mf


class PafResize(mf.core.MakiLayer):
    def __init__(self, size=None, name='PafResize'):
        self.size = size
        super().__init__(name=name, params=[], regularize_params=[], named_params_dict={})

    def forward(self, x, computation_mode=mf.core.MakiLayer.INFERENCE_MODE):
        if isinstance(x, list):
            paf, size = x
        else:
            paf, size = x, self.size

        shape_paf = tf.shape(paf)

        # [N, H, W, NUM_PAFS, 2] --> [N, H, W, NUM_PAFS * 2]
        paf = tf.reshape(
            paf,
            shape=tf.stack([shape_paf[0], shape_paf[1], shape_paf[2], -1], axis=0)
        )

        resized_paf = tf.image.resize_nearest_neighbor(
            paf,
            size,
            align_corners=False,
            name='upsample_paf'
        )

        return resized_paf

    def training_forward(self, x):
        return self.forward(x)

    @staticmethod
    def build(params: dict):
        pass

    def to_dict(self):
        pass
