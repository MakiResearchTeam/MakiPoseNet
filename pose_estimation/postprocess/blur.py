import numpy as np
import scipy.stats as st
import tensorflow as tf

import makiflow as mf


class Blur(mf.core.MakiLayer):
    DATA = 'data'
    SMOOTHER_OP_NAME = 'smoother_op'
    SMOOTHER_VAR_NAME = 'smoother_kernel'

    def __init__(self, filter_size, sigma, heat_map_size=0, preload_kernel=None, dtype=np.float32, name='blur'):
        self.filter_size = filter_size
        self.sigma = sigma
        self.heat_map_size = heat_map_size

        self._smoother_kernel = None
        self._smoother_op = None
        self.setup_kernel(
            filter_size=filter_size, sigma=sigma,
            heat_map_size=heat_map_size, preload_kernel=preload_kernel, dtype=dtype
        )
        super().__init__(name=name, params=[], regularize_params=[], named_params_dict={})

    def setup_kernel(self, filter_size, sigma, heat_map_size, preload_kernel, dtype):
        if preload_kernel is None:
            kernel = self.gauss_kernel(filter_size, sigma, heat_map_size)
        else:
            print('Use preload kernel for smoother')
            kernel = preload_kernel

        self._smoother_kernel = tf.constant(
            kernel.astype(dtype),
            name=Blur.SMOOTHER_VAR_NAME,
            dtype=tf.dtypes.as_dtype(dtype)
        )

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        return out_filter

    def forward(self, x, computation_mode=mf.core.MakiLayer.INFERENCE_MODE):
        return tf.nn.depthwise_conv2d(
            x, self._smoother_kernel,
            [1, 1, 1, 1],
            padding='SAME', name=Blur.SMOOTHER_OP_NAME
        )

    def training_forward(self, x):
        return self.forward(x)

    @staticmethod
    def build(params: dict):
        pass

    def to_dict(self):
        pass

    @property
    def kernel(self):
        return self._smoother_kernel
