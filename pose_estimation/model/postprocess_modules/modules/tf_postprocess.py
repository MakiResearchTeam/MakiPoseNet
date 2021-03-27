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
from pose_estimation.model.utils.smoother import Smoother


class TFPostProcessModule(InterfacePostProcessModule):

    UPSAMPLE_SIZE = 'upsample_size'

    _DEFAULT_KERNEL_MAX_POOL = [1, 3, 3, 1]

    def __init__(self, smoother_kernel_size=20, smoother_kernel=None, fast_mode=False,
            prediction_up_scale=8, upsample_heatmap_after_down_scale=False, ignore_last_dim_inference=True,
            threash_hold_peaks=0.1, use_blur=True, heatmap_resize_method=tf.image.resize_nearest_neighbor,
            second_heatmap_resize_method=tf.image.resize_bilinear):
        """

        Parameters
        ----------
        smoother_kernel_size : int
            Size of a kernel in the smoother (aka gaussian filter)
        smoother_kernel : np.ndarray
            Size of the kernel window for smoother
            Smoother always use square of size [`smoother_kernel`, `smoother_kernel`] as kernel
        fast_mode : bool
            If equal to True, max_pool operation will be change to a binary operations,
            which are faster, but can give lower accuracy
        prediction_up_scale : int
            In most scenarios PEModel gives heatmap with size lower by factor 8 compare to input image size,
            So heatmap will be scaled (of resized) to original size by default,
            i.e. `prediction_up_scale` = 8,
            But heatmap can be resized by factor 4, and on this size peaks/indices can be taken,
            This method will be much faster, but can gives lower accuracy
        upsample_heatmap_after_down_scale : bool
            As mentioned in `prediction_up_scale`, heatmap can have size not equal to original input image,
            In order to speed up inference, smoother used at size mentioned in `prediction_up_scale` parameter
            And after that (smoothed) heatmap will be resized to original image size.
            This method can gives more accurate results, compare to previous method mentioned in `prediction_up_scale`
        ignore_last_dim_inference : bool
            In most models, last dimension is background heatmap and its does not used in inference
        threash_hold_peaks : float
            Threshold if keypoint have bigger probability, it will be used, otherwise it will as not detected keypoint
        heatmap_resize_method : tf.image
            Method of first resize for heatmap (to size mentioned in `prediction_up_scale`) in tf, f
            For more details, see tf.image docs
        second_heatmap_resize_method : tf.image
            Method of first resize for heatmap in tf
            (to size mentioned in `upsample_heatmap_after_down_scale`, i.e. smoothed heatmap sill be resized),
            For more details, see tf.image docs

        """
        super().__init__()
        self._smoother_kernel_size = smoother_kernel_size
        self._smoother_kernel = smoother_kernel
        self._fast_mode = fast_mode
        self._prediction_up_scale = prediction_up_scale

        self._upsample_heatmap_after_down_scale = upsample_heatmap_after_down_scale
        self._ignore_last_dim_inference = ignore_last_dim_inference
        self._threash_hold_peaks = threash_hold_peaks
        self._use_blur = use_blur

        self._heatmap_resize_method = heatmap_resize_method
        self._second_heatmap_resize_method = second_heatmap_resize_method
        self.upsample_size = None

    def get_indices_tensor(self) -> tf.Tensor:
        return self.__indices

    def get_peaks_score_tensor(self) -> tf.Tensor:
        return self.__peaks_score

    def get_peaks_tensor(self) -> tf.Tensor:
        return self._peaks

    def get_up_heatmap_tensor(self) -> tf.Tensor:
        return self._up_heatmap

    def get_resized_heatmap_tensor(self) -> tf.Tensor:
        return self._resized_heatmap

    def get_resized_paf_tensor(self) -> tf.Tensor:
        return self._resized_paf

    def get_smoother_output_heatmap_tensor(self) -> tf.Tensor:
        return self._blured_heatmap

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
        feed_dict.update({self.upsample_size: super().get_resize_to()})
        batched_paf, indices, peaks = self._session.run(
            [self._resized_paf, self.__indices, self.__peaks_score],
            feed_dict=feed_dict
        )
        return batched_paf[0], indices, peaks

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

        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2), name=TFPostProcessModule.UPSAMPLE_SIZE)
        self._build_paf_graph()
        self._build_heatmap_graph()
        self._build_peaks_graph()

    def _build_paf_graph(self):
        main_paf = super().get_paf_tensor()
        # [N, W, H, NUM_PAFS, 2]
        shape_paf = tf.shape(main_paf)

        # [N, W, H, NUM_PAFS, 2] --> [N, NEW_W, NEW_H, NUM_PAFS * 2]
        main_paf = tf.reshape(
            main_paf,
            shape=tf.stack([shape_paf[0], shape_paf[1], shape_paf[2], -1], axis=0)
        )
        self._resized_paf = tf.image.resize_nearest_neighbor(
            main_paf,
            self.upsample_size,
            align_corners=False,
            name='upsample_paf'
        )

    def _build_heatmap_graph(self):
        main_heatmap = super().get_heatmap_tensor()
        if self._ignore_last_dim_inference:
            main_heatmap = main_heatmap[..., :-1]

        # For more faster inference,
        # We can take peaks on low res heatmap
        if self._prediction_up_scale != TFPostProcessModule.DEFAULT_SCALE:
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

        if self._upsample_heatmap_after_down_scale:
            self._up_heatmap = self._second_heatmap_resize_method(
                self._blured_heatmap,
                self.upsample_size,
                align_corners=False,
                name='second_upsample_heatmap'
            )
        else:
            self._up_heatmap = self._blured_heatmap

    def _build_peaks_graph(self):
        if self._fast_mode:
            heatmap_tf = self._up_heatmap[0]
            heatmap_tf = tf.where(
                tf.less(heatmap_tf, self._threash_hold_peaks),
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

            peaks = (heatmap_center > heatmap_left)  & \
                    (heatmap_center > heatmap_right) & \
                    (heatmap_center > heatmap_up)    & \
                    (heatmap_center > heatmap_down)
            self._peaks = tf.cast(peaks, tf.float32)
        else:
            # Apply NMS (Non maximum suppression)
            # Apply max pool operation to heatmap
            max_pooled_heatmap = tf.nn.max_pool(
                self._up_heatmap,
                TFPostProcessModule._DEFAULT_KERNEL_MAX_POOL,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            # Take only values that equal to heatmap from max pooling,
            # i.e. biggest numbers of heatmaps
            self._peaks = tf.where(
                tf.equal(
                    self._up_heatmap,
                    max_pooled_heatmap
                ),
                self._up_heatmap,
                tf.zeros_like(self._up_heatmap)
            )[0]

        self.__indices, self.__peaks_score = self.__get_peak_indices_tf(self._peaks, thresh=self._threash_hold_peaks)

        if self._prediction_up_scale != TFPostProcessModule.DEFAULT_SCALE and not self._upsample_heatmap_after_down_scale:
            # indices - [num_indx, 3], first two dimensions - xy,
            # last dims - keypoint class (in order to save keypoint class scale to 1)
            scale = int(round(float(TFPostProcessModule.DEFAULT_SCALE) / self._prediction_up_scale))
            scale_kp = np.array([scale] * 2 + [1], dtype=np.int32)
            self.__indices = self.__indices * scale_kp

    def __get_peak_indices_tf(self, array: tf.Tensor, thresh=0.1):
        """
        Returns array indices of the values larger than threshold.

        Parameters
        ----------
        array : ndarray of any shape
            Tensor which values' indices to gather.
        thresh : float
            Threshold value.

        Returns
        -------
        ndarray of shape [n_peaks, dim(array)]
            Array of indices of the values larger than threshold.
        ndarray of shape [n_peaks]
            Array of the values at corresponding indices.

        """
        flat_peaks = tf.reshape(array, [-1])
        coords = tf.range(0, tf.shape(flat_peaks)[0], dtype=tf.int32)

        peaks_coords = coords[flat_peaks > thresh]

        peaks = tf.gather(flat_peaks, peaks_coords)

        indices = tf.transpose(tf.unravel_index(peaks_coords, dims=tf.shape(array)), [1, 0])
        return indices, peaks

    def get_data_for_debug(self, feed_dict):
        """
        Gives peaks, heatmap and paf batched tensors according to input `feed_dict`
        Usually this method are used for debug purposes and not recommended for other stuff

        Parameters
        ----------
        feed_dict : dict
            Example: { placholder: np.ndarray }, which futher calls with session

        Returns
        -------
        peaks : np.ndarray
        heatmap : np.ndarray
        paf : np.ndarray

        """
        super()._check_graph()
        resize_to = super().get_resize_to()
        feed_dict.update({self.upsample_size: resize_to})

        batched_paf, batched_heatmap, batched_peaks = self._session.run(
            [self._resized_paf, self._up_heatmap, self._peaks],
            feed_dict=feed_dict
        )

        # For paf
        # [N, NEW_W, NEW_H, NUM_PAFS * 2]
        shape_paf = batched_paf.shape
        N = shape_paf[0]
        num_pafs = shape_paf[-1] // 2
        # [N, NEW_W, NEW_H, NUM_PAFS * 2] --> [N, NEW_W, NEW_H, NUM_PAFS, 2]
        return batched_peaks, batched_heatmap, batched_paf.reshape(N, *resize_to, num_pafs, 2)
