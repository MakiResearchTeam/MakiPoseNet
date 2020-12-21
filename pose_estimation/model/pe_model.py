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

import json
import numpy as np
import tensorflow as tf

from .core import PoseEstimatorInterface
from .utils.algorithm_connect_skelet import estimate_paf, merge_similar_skelets
from .utils.smoother import Smoother
from makiflow.core import MakiTensor, MakiModel
from makiflow.core.inference.maki_builder import MakiBuilder


class PEModel(PoseEstimatorInterface):
    INPUT_MT = 'input_mt'
    OUTPUT_HEATMAP_MT = 'output_heatmap_mt'
    OUTPUT_PAF_MT = 'output_paf_mt'
    NAME = 'name'

    UPSAMPLE_SIZE = 'upsample_size'

    _DEFAULT_KERNEL_MAX_POOL = [1, 3, 3, 1]

    @staticmethod
    def from_json(
            path_to_model: str, input_tensor: MakiTensor = None,
            smoother_kernel_size=25, fast_mode=False,
            prediction_down_scale=1, ignore_last_dim_inference=True,
            threash_hold_peaks=0.1):
        """
        Creates and returns PEModel from json file contains its architecture

        Parameters
        ----------
        path_to_model : str
            Path to model which are saved as json file.
            Example: /home/user/model.json
        input_tensor : MakiTensor
            Custom input tensor for model, in most cases its just placeholder.
            By default equal to None, i.e. placeholder as input for model will be created automatically
        fast_mode : bool
            If equal to True, max_pool operation will be change to a binary operations,
            which are faster, but can give lower accuracy
        prediction_down_scale : int
            At which scale build skeletons,
            If more than 1, final keypoint will be scaled to size of the input image
        smoother_kernel_size : int
            Size of a kernel in the smoother (aka gaussian filter)
        ignore_last_dim_inference : bool
            In most models, last dimension is background heatmap and its does not used in inference
        threash_hold_peaks : float
            pass

        """
        # Read architecture from file
        json_file = open(path_to_model)
        json_value = json_file.read()
        json_file.close()

        json_info = json.loads(json_value)

        # Take model information
        output_heatmap_mt_names = json_info[MakiModel.MODEL_INFO][PEModel.OUTPUT_HEATMAP_MT]
        output_paf_mt_names = json_info[MakiModel.MODEL_INFO][PEModel.OUTPUT_PAF_MT]

        input_mt_name = json_info[MakiModel.MODEL_INFO][PEModel.INPUT_MT]
        model_name = json_info[MakiModel.MODEL_INFO][PEModel.NAME]
        graph_info = json_info[MakiModel.GRAPH_INFO]

        # Restore all graph variables of saved model
        inputs_and_outputs = MakiBuilder.restore_graph(
            output_heatmap_mt_names + output_paf_mt_names,
            graph_info,
            input_layer=input_tensor
        )

        input_x = input_tensor
        if input_x is None:
            input_x = inputs_and_outputs[input_mt_name]

        output_paf_list = [inputs_and_outputs[take_by_name] for take_by_name in output_paf_mt_names]
        output_heatmap_list = [inputs_and_outputs[take_by_name] for take_by_name in output_heatmap_mt_names]

        print('Model is restored!')

        return PEModel(
            input_x=input_x,
            output_heatmap_list=output_heatmap_list,
            output_paf_list=output_paf_list,
            smoother_kernel_size=smoother_kernel_size,
            fast_mode=fast_mode,
            prediction_down_scale=prediction_down_scale,
            ignore_last_dim_inference=ignore_last_dim_inference,
            threash_hold_peaks=threash_hold_peaks,
            name=model_name
        )

    @staticmethod
    def build_inference_part(
            main_paf: tf.Tensor, main_heatmap: tf.Tensor,
            fast_mode=False, prediction_down_scale=1,
            smoother_kernel_size=25, threash_hold_peaks=0.1):
        """
        Build inference part of the graph

        Parameters
        ----------
        main_paf : tf.Tensor
            Main (final) paf tensor
            Shape: [N, H, W, NUM_KP // 2, 2]
        main_heatmap : tf.Tensor
            Main (final) heatmap tensor
            Shape: [N, H, W, NUM_KP]
        fast_mode : bool
            If equal to True, max_pool operation will be change to a binary operations,
            which are faster, but can give lower accuracy
        prediction_down_scale : int
            At which scale build skeletons,
            If more than 1, final keypoint will be scaled to size of the input image
        smoother_kernel_size : int
            Size of a kernel in the smoother (aka gaussian filter)
        threash_hold_peaks : float
            Threash hold for peaks

        Returns
        -------
        upsample_size : tf.placeholder
            Placeholder, final heatmap and paf size after resize op
            Shape: [2]
        smoother : Smoother obj
            Gaussian filter obj
        resized_paf : tf.Tensor
            Resized paf to upsample_size size
            Shape: [N, upsample_size[0], upsample_size[1], NUM_KP // 2, 2]
        resized_heatmap : tf.Tensor
            Resized heatmap to upsample_size size
            Shape: [N, upsample_size[0], upsample_size[1], NUM_KP ]
        peaks : tf.Tensor
            Peaks of the predictions
            Shape: [N, W, H, NUM_KP]
        indices : tf.Tensor
            Indices of the maximum values on the peaks, store x,y coordinate and keypoint class (int)
            Shape: [N, num_ind, 3]
        peaks_score : tf.Tensor
            Score for every indices
            Shape: [N, num_ind]

        """
        if not isinstance(prediction_down_scale, int):
            raise TypeError(f"Parameter: `prediction_down_scale` should have type int, "
                            f"but type:`{type(prediction_down_scale)}` were given with value: {prediction_down_scale}.")

        if not isinstance(smoother_kernel_size, int) or smoother_kernel_size <= 0 or smoother_kernel_size >= 50:
            raise TypeError(f"Parameter: `smoother_kernel_size` should have type int "
                            f"and this value should be in range (0, 50), "
                            f"but type:`{type(smoother_kernel_size)}` were given with value: {smoother_kernel_size}.")

        if threash_hold_peaks <= 0.0 or threash_hold_peaks >= 1.0:
            raise TypeError(f"Parameter: `threash_hold_peaks` should be in range (0.0, 1.0), "
                            f"but value: {threash_hold_peaks} were given.")

        # Store (H, W) - final size of the prediction
        upsample_size = tf.placeholder(dtype=tf.int32, shape=(2), name=PEModel.UPSAMPLE_SIZE)

        # [N, W, H, NUM_PAFS, 2]
        shape_paf = tf.shape(main_paf)

        # [N, W, H, NUM_PAFS * 2] --> [N, NEW_W, NEW_H, NUM_PAFS * 2]
        main_paf = tf.reshape(
            main_paf,
            shape=tf.stack([shape_paf[0], shape_paf[1], shape_paf[2], -1], axis=0)
        )
        resized_paf = tf.image.resize_nearest_neighbor(
            main_paf,
            upsample_size,
            align_corners=False,
            name='upsample_paf'
        )

        # For more faster inference,
        # We can take peaks on low res heatmap
        if prediction_down_scale > 1:
            final_heatmap_size = upsample_size // prediction_down_scale
        else:
            final_heatmap_size = upsample_size

        resized_heatmap = tf.image.resize_nearest_neighbor(
            main_heatmap,
            final_heatmap_size,
            align_corners=False,
            name='upsample_heatmap'
        )

        num_keypoints = main_heatmap.get_shape().as_list()[-1]
        smoother = Smoother(
            {Smoother.DATA: resized_heatmap},
            smoother_kernel_size,
            3.0,
            num_keypoints
        )

        if fast_mode:
            heatmap_tf = smoother.get_output()[0]
            heatmap_tf = tf.where(
                tf.less(heatmap_tf, threash_hold_peaks),
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
            peaks = tf.cast(peaks, tf.float32)
        else:
            # Apply NMS (Non maximum suppression)
            # Apply max pool operation to heatmap
            max_pooled_heatmap = tf.nn.max_pool(
                smoother.get_output(),
                PEModel._DEFAULT_KERNEL_MAX_POOL,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            # Take only values that equal to heatmap from max pooling,
            # i.e. biggest numbers of heatmaps
            peaks = tf.where(
                tf.equal(
                    smoother.get_output(),
                    max_pooled_heatmap
                ),
                smoother.get_output(),
                tf.zeros_like(smoother.get_output())
            )[0]

        indices, peaks_score = PEModel._get_peak_indices_tf(peaks, thresh=threash_hold_peaks)

        if prediction_down_scale > 1:
            # indices - [num_indx, 3], first two dimensions - xy,
            # last dims - keypoint class (in order to save keypoint class scale to 1)
            indices = indices * np.array([prediction_down_scale]*2 +[1], dtype=np.int32)

        return [
            upsample_size, smoother,
            resized_paf, resized_heatmap,
            peaks, indices, peaks_score
        ]

    @staticmethod
    def _get_peak_indices_tf(array: tf.Tensor, thresh=0.1):
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

    def __init__(
        self,
        input_x: MakiTensor,
        output_paf_list: list,
        output_heatmap_list: list,
        smoother_kernel_size=25,
        ignore_last_dim_inference=True,
        prediction_down_scale=1,
        fast_mode=False,
        threash_hold_peaks=0.1,
        name="Pose_estimation"
    ):
        """
        Create Pose Estimation Model which provides API to train and tests model.

        Parameters
        ----------
        input_x : MakiTensor
            Input MakiTensor
        output_paf_list : list
            List of MakiTensors which are output part affinity fields (paf).
            Assume that last tensor in the list, will be the main one
        output_heatmap_list : list
            List of MakiTensors which are output heatmaps.
            Assume that last tensor in the list, will be the main one
        smoother_kernel_size : int
            Size of a kernel in the smoother (aka gaussian filter)
        ignore_last_dim_inference : bool
            In most models, last dimension is background heatmap and its does not used in inference
        prediction_down_scale : int
            At which scale build skeletons,
            If more than 1, final keypoint will be scaled to size of the input image
        threash_hold_peaks : float
            pass
        fast_mode : bool
            If equal to True, max_pool operation will be change to a binary operations,
            which are faster, but can give lower accuracy
        name : str
            Name of this model
        """
        self.name = str(name)
        self._in_x = input_x
        self._paf_list = output_paf_list
        self._heatmap_list = output_heatmap_list
        self._index_of_main_paf = len(output_paf_list) - 1
        super().__init__(outputs=output_paf_list + output_heatmap_list, inputs=[input_x])
        self._init_tensors_for_prediction(
            smoother_kernel_size=smoother_kernel_size,
            ignore_last_dim_inference=ignore_last_dim_inference,
            prediction_down_scale=prediction_down_scale,
            fast_mode=fast_mode,
            threash_hold_peaks=threash_hold_peaks
        )

    def _init_tensors_for_prediction(
            self,
            smoother_kernel_size,
            ignore_last_dim_inference,
            prediction_down_scale,
            fast_mode,
            threash_hold_peaks):
        """
        Initialize tensors for prediction

        """
        self.__saved_mesh_grid = None
        main_heatmap = self.get_main_heatmap_tensor()
        if ignore_last_dim_inference:
            main_heatmap = main_heatmap[..., :-1]

        main_paf = self.get_main_paf_tensor()

        tensors = PEModel.build_inference_part(
            main_paf, main_heatmap,
            smoother_kernel_size=smoother_kernel_size,
            prediction_down_scale=prediction_down_scale,
            fast_mode=fast_mode,
            threash_hold_peaks=threash_hold_peaks
        )
        self.upsample_size = tensors[0]
        self._smoother = tensors[1]
        self._resized_paf = tensors[2]
        self._resized_heatmap = tensors[3]
        self._peaks = tensors[4]
        self.__indices = tensors[5]
        self.__peaks_score = tensors[6]

    def set_session(self, session: tf.Session):
        super().set_session(session)

    def predict(self, x: list, resize_to=None, using_estimate_alg=True):
        """
        Do pose estimation on certain input images

        Parameters
        ----------
        x : list or np.ndarray
            Input list/np.ndarray of the images
        resize_to : tuple
            Tuple of two int [H, W], which are size of the output. H - Height, W - Width.
            Resize prediction from neural network to certain size.
            By default resize not be used. If it used, by default used area interpolation
        using_estimate_alg : bool
            If equal True, when algorithm to build skeletons will be used
            And method will return list of the class Human (See Return for more detail)
            NOTICE! If equal True, then only first batch size (i.e. with batch_size = 1) will be processed.
            Otherwise, method will return peaks, heatmap and paf

        Returns
        -------
        if using_estimate_alg is True:
            list
                List of predictions to each input image.
                NOTICE! Only first batch size (i.e. with batch_size = 1) will be processed.
                Single element of this list is a List of classes Human which were detected.

        Otherwise:
            np.ndarray
                Peaks
            np.ndarray
                Heatmap
            np.ndarray
                Paf
        """
        # Take predictions
        if resize_to is None:
            # Take `H`, `W` from input image
            resize_to = x[0].shape[:2]

        if using_estimate_alg:

            batched_paf, indices, peaks = self._session.run(
                [self._resized_paf, self.__indices, self.__peaks_score],
                feed_dict={
                    self._input_data_tensors[0]: x,
                    self.upsample_size: resize_to
                }
            )

            return [
                merge_similar_skelets(estimate_paf(
                    peaks=peaks.astype(np.float32, copy=False),
                    indices=indices.astype(np.int32, copy=False),
                    paf_mat=batched_paf[0]
                ))
            ]

        else:
            batched_paf, batched_heatmap, batched_peaks = self._session.run(
                [self._resized_paf, self._smoother.get_output(), self._peaks],
                feed_dict={
                    self._input_data_tensors[0]: x,
                    self.upsample_size: resize_to
                }
            )

            # For paf
            # [N, NEW_W, NEW_H, NUM_PAFS * 2]
            shape_paf = batched_paf.shape
            N = shape_paf[0]
            num_pafs = shape_paf[-1] // 2
            # [N, NEW_W, NEW_H, NUM_PAFS * 2] --> [N, NEW_W, NEW_H, NUM_PAFS, 2]
            return batched_peaks, batched_heatmap, batched_paf.reshape(N, *resize_to, num_pafs, 2)

    def __get_peak_indices(self, array, thresh=0.1):
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
        flat_peaks = np.reshape(array, -1)
        if self.__saved_mesh_grid is None or len(flat_peaks) != self.__saved_mesh_grid.shape[0]:
            self.__saved_mesh_grid = np.arange(len(flat_peaks))

        peaks_coords = self.__saved_mesh_grid[flat_peaks > thresh]

        peaks = flat_peaks.take(peaks_coords)

        indices = np.unravel_index(peaks_coords, shape=array.shape)
        indices = np.stack(indices, axis=-1)
        return indices, peaks

    def get_indices_tensor(self) -> tf.Tensor:
        return self.__indices

    def get_peaks_score_tensor(self) -> tf.Tensor:
        return self.__peaks_score

    def get_peaks_tensor(self) -> tf.Tensor:
        return self._peaks

    def get_resized_paf_tensor(self) -> tf.Tensor:
        return self._resized_paf

    def get_smoother_output_heatmap_tensor(self) -> tf.Tensor:
        return self._smoother.get_output()

    def get_main_paf_tensor(self):
        return self._output_data_tensors[self._index_of_main_paf]

    def get_main_heatmap_tensor(self):
        return self._output_data_tensors[-1]

    def get_paf_makitensors(self):
        """
        Return list of mf.MakiTensors which are the paf (party affinity fields) calculation tensor
        """
        return self._paf_list

    def get_heatmap_makitensors(self):
        """
        Return list of mf.MakiTensor which are the heatmap calculation tensor
        """
        return self._heatmap_list

    def get_batch_size(self):
        """
        Return batch size
        """
        return self._inputs[0].get_shape()[0]

    def get_feed_dict_config(self) -> dict:
        return {
            self._in_x: 0
        }

    def _get_model_info(self):
        """
        Return information about model for saving architecture in JSON file
        """
        input_mt = self._inputs[0]
        output_heatmap_mt_names = [elem.get_name() for elem in self.get_heatmap_makitensors()]
        output_paf_mt_names = [elem.get_name() for elem in self.get_paf_makitensors()]

        return {
            PEModel.INPUT_MT: input_mt.get_name(),
            PEModel.OUTPUT_HEATMAP_MT: output_heatmap_mt_names,
            PEModel.OUTPUT_PAF_MT: output_paf_mt_names,
            PEModel.NAME: self.name
        }

