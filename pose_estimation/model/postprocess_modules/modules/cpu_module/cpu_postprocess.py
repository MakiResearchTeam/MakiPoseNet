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

import tensorflow as tf
import numpy as np
from pose_estimation.model.postprocess_modules.core import InterfacePostProcessModule
from .cpu_postprocess_np_part import CPUOptimizedPostProcessNPPart
from pose_estimation.model.utils.smoother import Smoother


class CPUOptimizedPostProcessModule(InterfacePostProcessModule):

    DEFAULT_SCALE = 8

    UPSAMPLE_SIZE = 'upsample_size'

    _DEFAULT_KERNEL_MAX_POOL = [1, 3, 3, 1]

    def __init__(self, smoother_kernel_size=25, smoother_kernel=None,
            prediction_up_scale=8, upsample_heatmap_after_down_scale=False, ignore_last_dim_inference=True,
            threash_hold_peaks=0.1, use_blur=True, heatmap_resize_method=tf.image.resize_nearest_neighbor):
        """

        Parameters
        ----------
        smoother_kernel_size : int
            Size of a kernel in the smoother (aka gaussian filter)
        smoother_kernel : np.ndarray
            TODO: Add docs
        prediction_up_scale : int
            TODO: ADD docs
        upsample_heatmap_after_down_scale : bool
            TODO: ADD docs
        ignore_last_dim_inference : bool
            In most models, last dimension is background heatmap and its does not used in inference
        threash_hold_peaks : float
            pass
        heatmap_resize_method : tf.image
            TODO: add docs

        """
        super().__init__()
        self._smoother_kernel_size = smoother_kernel_size
        self._smoother_kernel = smoother_kernel
        self._prediction_up_scale = prediction_up_scale

        self._upsample_heatmap_after_down_scale = upsample_heatmap_after_down_scale
        self._ignore_last_dim_inference = ignore_last_dim_inference
        self._threash_hold_peaks = threash_hold_peaks
        self._use_blur = use_blur

        self._heatmap_resize_method = heatmap_resize_method

        self.upsample_size = None
        self._postprocess_np_tools = None

    def set_resize_to(self, resize_to: tuple):
        if self._postprocess_np_tools is not None:
            self._postprocess_np_tools.set_resize_to(resize_to)
        super().set_resize_to(resize_to)

    def _execute_postprocess(self, feed_dict):
        """
        Execute tf graph of postprocess and model itself according to input feed_dict

        Parameters
        ----------
        feed_dict : dict
            Example: { placholder: np.ndarray }, which further calls with session

        Returns
        -------
        paf : np.ndarray
        indices : np.ndarray
        peaks : np.ndarray

        """
        resize_to = super().get_resize_to()
        feed_dict.update({self.upsample_size: resize_to})

        paf_pr, smoothed_heatmap_pr = self._session.run(
            [super().get_paf_tensor(), self._blured_heatmap],
            feed_dict=feed_dict
        )
        # Other part of execution graph are written in numpy/cv style
        # Because they are much faster on CPU, than operation through TF on CPU
        return self._postprocess_np_tools.process(smoothed_heatmap_pr, paf_pr)

    def _build_postporcess_graph(self):
        """
        Initialize tensors for prediction

        """
        if not isinstance(self._prediction_up_scale, int):
            raise TypeError(f"Parameter: `_prediction_up_scale` should have type int, "
                            f"but type:`{type(self._prediction_up_scale)}` "
                            f"were given with value: {self._prediction_up_scale}.")

        if self._prediction_up_scale <= 0 or self._prediction_up_scale > 8:
            raise ValueError(f"Parameter: `_prediction_up_scale` must be in range (0, 8], "
                             f"but its value is:`{self._prediction_up_scale}`")

        if not isinstance(self._smoother_kernel_size, int) or \
                self._smoother_kernel_size <= 0 or self._smoother_kernel_size >= 50:
            raise TypeError(f"Parameter: `smoother_kernel_size` should have type int "
                            f"and this value should be in range (0, 50), "
                            f"but type:`{type(self._smoother_kernel_size)}`"
                            f" were given with value: {self._smoother_kernel_size}.")

        if self._threash_hold_peaks <= 0.0 or self._threash_hold_peaks >= 1.0:
            raise TypeError(f"Parameter: `threash_hold_peaks` should be in range (0.0, 1.0), "
                            f"but value: {self._threash_hold_peaks} were given.")

        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2), name=CPUOptimizedPostProcessModule.UPSAMPLE_SIZE)
        self._build_heatmap_graph()

        kp_scale = None
        if not self._upsample_heatmap_after_down_scale:
            scale = int(round(float(CPUOptimizedPostProcessModule.DEFAULT_SCALE) / self._prediction_up_scale))
            kp_scale = np.array([scale] * 2 + [1], dtype=np.int32)

        self._postprocess_np_tools = CPUOptimizedPostProcessNPPart(
            super().get_resize_to(),
            self._upsample_heatmap_after_down_scale,
            kp_scale_end=kp_scale
        )

    def _build_heatmap_graph(self):
        main_heatmap = super().get_heatmap_tensor()
        if self._ignore_last_dim_inference:
            main_heatmap = main_heatmap[..., :-1]

        # For more faster inference,
        # We can take peaks on low res heatmap
        if self._prediction_up_scale != CPUOptimizedPostProcessModule.DEFAULT_SCALE:
            heatmap_shape = tf.shape(main_heatmap)[1:3] # H, W
            final_heatmap_size = heatmap_shape * self._prediction_up_scale
        else:
            final_heatmap_size = self.upsample_size

        self._resized_heatmap = self._heatmap_resize_method(
            main_heatmap,
            final_heatmap_size,
            align_corners=False,
            name='upsample_heatmap'
        )
        num_keypoints = main_heatmap.get_shape().as_list()[-1]

        if self._use_blur:
            self._smoother = Smoother(
                self._resized_heatmap,
                self._smoother_kernel_size,
                3.0,
                num_keypoints,
                preload_kernel=self._smoother_kernel
            )
            self._blured_heatmap = self._smoother.get_output()
        else:
            self._blured_heatmap = self._resized_heatmap

    def get_resized_heatmap_tensor(self) -> tf.Tensor:
        return self._resized_heatmap

    def get_smoother_output_heatmap_tensor(self) -> tf.Tensor:
        return self._blured_heatmap

