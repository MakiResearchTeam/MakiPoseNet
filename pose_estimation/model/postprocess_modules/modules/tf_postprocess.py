import tensorflow as tf
import numpy as np
from pose_estimation.model.postprocess_modules.core.postprocess import InterfacePostProcessModule
from pose_estimation.model.utils.algorithm_connect_skelet import merge_similar_skelets, estimate_paf
from pose_estimation.model.utils.smoother import Smoother


class TFPostProcessModule(InterfacePostProcessModule):

    UPSAMPLE_SIZE = 'upsample_size'

    _DEFAULT_KERNEL_MAX_POOL = [1, 3, 3, 1]

    def __init__(self, smoother_kernel_size=25, smoother_kernel=None, fast_mode=False,
            prediction_down_scale=1, upsample_heatmap_after_down_scale=False, ignore_last_dim_inference=True,
            threash_hold_peaks=0.1, use_blur=True, heatmap_resize_method=tf.image.resize_nearest_neighbor,
            second_heatmap_resize_method=tf.image.resize_bilinear):
        """

        Parameters
        ----------
        smoother_kernel_size : int
            Size of a kernel in the smoother (aka gaussian filter)
        smoother_kernel : np.ndarray
            TODO: Add docs
        fast_mode : bool
            If equal to True, max_pool operation will be change to a binary operations,
            which are faster, but can give lower accuracy
        prediction_down_scale : int
            At which scale build skeletons,
            If more than 1, final keypoint will be scaled to size of the input image
        upsample_heatmap_after_down_scale : bool
            TODO: ADD docs
        ignore_last_dim_inference : bool
            In most models, last dimension is background heatmap and its does not used in inference
        threash_hold_peaks : float
            pass
        heatmap_resize_method : tf.image
            TODO: add docs
        second_heatmap_resize_method : tf.image
            TODO: add docs

        """
        self._smoother_kernel_size = smoother_kernel_size
        self._smoother_kernel = smoother_kernel

        self._fast_mode = fast_mode
        self._prediction_down_scale = prediction_down_scale

        self._upsample_heatmap_after_down_scale = upsample_heatmap_after_down_scale
        self._ignore_last_dim_inference = ignore_last_dim_inference

        self._threash_hold_peaks = threash_hold_peaks
        self._use_blur = use_blur

        self._heatmap_resize_method = heatmap_resize_method
        self._second_heatmap_resize_method = second_heatmap_resize_method

        self._heatmap_tensor = None
        self._paf_tensor = None
        self._session = None
        self._is_graph_build = False
        self._is_using_estimate_alg = True

    def set_session(self, session):
        self._session = session

    def set_is_using_estimate_alg(self, is_use: bool):
        self._is_using_estimate_alg = is_use

    def set_paf_heatmap(self, paf, heatmap):
        self._paf_tensor = paf
        self._heatmap_tensor = heatmap

    def __call__(self, input_batch, feed_dict):
        """

        Parameters
        ----------
        feed_dict : dict
            Example: { placholder: np.ndarray }, which futher calls with session

        Returns
        -------
        paf, indices, peaks

        """
        if not self._is_graph_build:
            self._build_postporcess_graph()
        resize_to = input_batch[0].shape[:2]
        feed_dict.update({self.upsample_size: resize_to})

        if self._is_using_estimate_alg:
            batched_paf, indices, peaks = self._session.run(
                [self._resized_paf, self.__indices, self.__peaks_score],
                feed_dict=feed_dict
            )
            return self._process_not_tf_part(batched_paf, indices, peaks)

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

    def _process_not_tf_part(self, batched_paf, indices, peaks):
        return [
                merge_similar_skelets(estimate_paf(
                    peaks=peaks.astype(np.float32, copy=False),
                    indices=indices.astype(np.int32, copy=False),
                    paf_mat=batched_paf[0]
                ))
        ]

    def _build_postporcess_graph(self):
        """
        Initialize tensors for prediction

        """
        main_heatmap = self._heatmap_tensor
        if self._ignore_last_dim_inference:
            main_heatmap = main_heatmap[..., :-1]

        main_paf = self._paf_tensor

        if not isinstance(self._prediction_down_scale, int):
            raise TypeError(f"Parameter: `prediction_down_scale` should have type int, "
                            f"but type:`{type(self._prediction_down_scale)}` "
                            f"were given with value: {self._prediction_down_scale}.")

        if not isinstance(self._smoother_kernel_size, int) or \
                self._smoother_kernel_size <= 0 or self._smoother_kernel_size >= 50:
            raise TypeError(f"Parameter: `smoother_kernel_size` should have type int "
                            f"and this value should be in range (0, 50), "
                            f"but type:`{type(self._smoother_kernel_size)}`"
                            f" were given with value: {self._smoother_kernel_size}.")

        if self._threash_hold_peaks <= 0.0 or self._threash_hold_peaks >= 1.0:
            raise TypeError(f"Parameter: `threash_hold_peaks` should be in range (0.0, 1.0), "
                            f"but value: {self._threash_hold_peaks} were given.")

        # Store (H, W) - final size of the prediction
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2), name=TFPostProcessModule.UPSAMPLE_SIZE)

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

        # For more faster inference,
        # We can take peaks on low res heatmap
        if self._prediction_down_scale > 1:
            final_heatmap_size = self.upsample_size // self._prediction_down_scale
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

        if self._prediction_down_scale > 1 and not self._upsample_heatmap_after_down_scale:
            # indices - [num_indx, 3], first two dimensions - xy,
            # last dims - keypoint class (in order to save keypoint class scale to 1)
            self.__indices = self.__indices * np.array([self._prediction_down_scale]*2 +[1], dtype=np.int32)

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

