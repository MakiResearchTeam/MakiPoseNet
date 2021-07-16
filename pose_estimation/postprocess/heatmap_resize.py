import tensorflow as tf

import makiflow as mf


class HeatmapResize(mf.core.MakiLayer):
    def __init__(
            self, size=None, resize_method=tf.image.resize_nearest_neighbor,
            ignore_last_dim=False, name='PafResize'
    ):
        self.size = size
        self.resize_method = resize_method
        self.ignore_last_dim = ignore_last_dim
        super().__init__(name=name, params=[], regularize_params=[], named_params_dict={})

    def forward(self, x, computation_mode=mf.core.MakiLayer.INFERENCE_MODE):
        if isinstance(x, list):
            heatmap, size = x
        else:
            heatmap, size = x, self.size

        if self.ignore_last_dim:
            heatmap = heatmap[..., :-1]

        resized_heatmap = self.resize_method(
            heatmap,
            size,
            align_corners=False,
            name='upsample_heatmap'
        )

        return resized_heatmap

    def training_forward(self, x):
        return self.forward(x)

    @staticmethod
    def build(params: dict):
        pass

    def to_dict(self):
        pass
