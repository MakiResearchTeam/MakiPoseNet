# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiPoseNet.
#
# MakiPoseNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiPoseNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

# vim: sta:et:sw=2:ts=2:sts=2
# Written by Antonio Loquercio

import numpy as np
import scipy.stats as st
import tensorflow as tf


class Smoother:

    DATA = 'data'
    SMOOTHER_OP_NAME = 'smoother_op'
    SMOOTHER_VAR_NAME = 'smoother_kernel'

    def __init__(self, inputs, filter_size, sigma, heat_map_size=0, preload_kernel=None, dtype=np.float32):
        self.inputs = inputs
        self.filter_size = filter_size
        self.sigma = sigma
        self.heat_map_size = heat_map_size

        self._smoother_kernel = None
        self._smoother_op = None
        self.setup_kernel(
            filter_size=filter_size, sigma=sigma,
            heat_map_size=heat_map_size, preload_kernel=preload_kernel, dtype=dtype
        )
        self.create_graph()

    def create_graph(self):
        self._smoother_op = tf.nn.depthwise_conv2d(
            self.inputs, self._smoother_kernel,
            [1, 1, 1, 1],
            padding='SAME', name=Smoother.SMOOTHER_OP_NAME
        )

    def setup_kernel(self, filter_size, sigma, heat_map_size, preload_kernel, dtype):
        if preload_kernel is None:
            kernel = self.gauss_kernel(filter_size, sigma, heat_map_size)
        else:
            print('Use preload kernel for smoother')
            kernel = preload_kernel

        self._smoother_kernel = tf.constant(
            kernel.astype(dtype),
            name=Smoother.SMOOTHER_VAR_NAME,
            dtype=tf.dtypes.as_dtype(dtype)
        )

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        out_filter = np.array(kernel, dtype = np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis = 2)
        return out_filter

    def get_output(self):
        """
        Returns the smoother output tf.Tensor

        """
        return self._smoother_op

    def get_variables(self):
        return [self._smoother_kernel]

