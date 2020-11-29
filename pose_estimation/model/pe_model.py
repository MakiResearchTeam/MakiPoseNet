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
import cv2

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

    def get_feed_dict_config(self) -> dict:
        return {
            self._in_x: 0
        }

    @staticmethod
    def from_json(path_to_model: str, input_tensor: MakiTensor = None):
        """
        Creates and returns PEModel from json file contains its architecture

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
            name=model_name
        )

    def __init__(
        self,
        input_x: MakiTensor,
        output_paf_list: list,
        output_heatmap_list: list,
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
        name : str
            Name of this model
        """
        self.name = str(name)
        self._in_x = input_x
        self._paf_list = output_paf_list
        self._heatmap_list = output_heatmap_list
        self._index_of_main_paf = len(output_paf_list) - 1
        super().__init__(outputs=output_paf_list + output_heatmap_list, inputs=[input_x])
        self._init_tensors_for_prediction()

    def _init_tensors_for_prediction(self):
        """
        Initialize tensors for prediction

        """
        num_keypoints = self.get_main_heatmap_tensor().get_shape().as_list()[-1]

        self.input_smoothed_image = tf.placeholder(
            shape=[1, None, None, num_keypoints],
            dtype=tf.float32,
            name='smoothed_input_image'
        )
        self._smoother = Smoother(
            {Smoother.DATA: self.input_smoothed_image},
            25,
            3.0,
            num_keypoints
        )

        # Apply NMS (Non maximum suppression)
        # Apply max pool operation to heatmap
        max_pooled_heatmap = tf.nn.max_pool(
            self._smoother.get_output(),
            self._DEFAULT_KERNEL_MAX_POOL,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        # Take only values that equal to heatmap from max pooling,
        # i.e. biggest numbers of heatmaps
        self._peaks = tf.where(
            tf.equal(
                self._smoother.get_output(),
                max_pooled_heatmap
            ),
            self._smoother.get_output(),
            tf.zeros_like(self._smoother.get_output())
        )

    def set_session(self, session: tf.Session):
        session.run(tf.variables_initializer(self._smoother.get_variables()))
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
            Otherwise, method will return peaks, heatmap and paf

        Returns
        -------
        if using_estimate_alg is True:
            list
                List of predictions to each input image.
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

        paf_prediction, heatmap_prediction = self._session.run(
            [self.get_main_paf_tensor(), self.get_main_heatmap_tensor()],
            feed_dict={self._input_data_tensors[0]: x}
        )

        # Process PAF
        # [N, W, H, NUM_PAFS, 2]
        shape_paf = paf_prediction.shape
        N, num_pafs = shape_paf[0], shape_paf[3]

        # [N, W, H, NUM_PAFS * 2] --> [N, NEW_W, NEW_H, NUM_PAFS * 2]
        paf_prediction_reshaped = paf_prediction.reshape(*shape_paf[:-2], -1)
        batched_paf = np.stack(
            [
                cv2.resize(paf_prediction_reshaped[i], (resize_to[1], resize_to[0]), interpolation=cv2.INTER_AREA)
                for i in range(len(paf_prediction_reshaped))
            ]
        )
        # Process HEATMAP
        batched_heatmap = np.stack(
            [
                cv2.resize(heatmap_prediction[i], (resize_to[1], resize_to[0]), interpolation=cv2.INTER_AREA)
                for i in range(len(heatmap_prediction))
            ]
        )
        # Get peaks
        batched_heatmap, batched_peaks = self._session.run(
            [self._smoother.get_output(), self._peaks],
            feed_dict={self.input_smoothed_image: batched_heatmap.astype(np.float32, copy=False)}
        )

        if using_estimate_alg:

            # Connect skeletons by applying two algorithms
            batched_humans = []

            for i in range(len(batched_peaks)):
                single_peaks = batched_peaks[i].astype(np.float32, copy=False)
                single_heatmap = batched_heatmap[i].astype(np.float32, copy=False)
                single_paff = batched_paf[i].astype(np.float32, copy=False)

                # Estimate
                humans_list = estimate_paf(single_peaks, single_heatmap, single_paff)
                # Remove similar points, simple merge similar skeletons
                humans_merged_list = merge_similar_skelets(humans_list)

                batched_humans.append(humans_merged_list)

            return batched_humans
        else:
            # For paf
            # [N, NEW_W, NEW_H, NUM_PAFS * 2] --> [N, NEW_W, NEW_H, NUM_PAFS, 2]
            return batched_peaks, batched_heatmap, batched_paf.reshape(N, *resize_to, num_pafs, 2)

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

