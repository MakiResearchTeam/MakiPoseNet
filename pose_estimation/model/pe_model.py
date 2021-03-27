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
from .postprocess_modules.core.postprocess import InterfacePostProcessModule
from .utils.skelet_builder import SkeletBuilder
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
            path_to_model: str, postprocess_class: InterfacePostProcessModule, input_tensor: MakiTensor = None):
        """
        Creates and returns PEModel from json file contains its architecture

        Parameters
        ----------
        path_to_model : str
            Path to model which are saved as json file.
            Example: /home/user/model.json
        postprocess_class : InterfacePostProcessModule
            # TODO: add docs
        input_tensor : MakiTensor
            Custom input tensor for model, in most cases its just placeholder.
            By default equal to None, i.e. placeholder as input for model will be created automatically

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
            postprocess_class=postprocess_class,
            name=model_name
        )

    def __init__(
        self,
        input_x: MakiTensor,
        output_paf_list: list,
        output_heatmap_list: list,
        postprocess_class: InterfacePostProcessModule,
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
        postprocess_class : InterfacePostProcessModule
            # TODO: add docs
        name : str
            Name of this model
        """
        self.name = str(name)
        self._in_x = input_x
        self._paf_list = output_paf_list
        self._heatmap_list = output_heatmap_list
        self._index_of_main_paf = len(output_paf_list) - 1
        self._postprocess_class = postprocess_class
        super().__init__(outputs=output_paf_list + output_heatmap_list, inputs=[input_x])

    def predict(self, image: np.ndarray, resize_to=None, using_estimate_alg=True):
        """
        Do pose estimation on certain input image

        Parameters
        ----------
        image : np.ndarray
            Input image as np.ndarray for model
            NOTICE! Input image must have dims equal to 3, otherwise the exception will be dropped
        resize_to : tuple
            Tuple of two int [H, W], which are size of the output. H - Height, W - Width.
            Resize prediction from neural network to certain size.
            By default resize not be used. If it used, by default used area interpolation
        using_estimate_alg : bool
            If equal True, when algorithm to build skeletons will be used
            And method will return list of the class Human (See Return for more detail)
            NOTICE! If equal True, then only first batch size (i.e. with batch_size = 1) will be processed.
            Otherwise, method will return peaks, heatmap and paf.
            This API usually are used for debug purposes

        Returns
        -------
        if using_estimate_alg is True:
            list
                List of classes Human which were detected on input image.
                NOTICE! Only batch size equal to 1 will be processed,
                i. e. If model batch size more than 1, then only first image will be process properly,
                And final predictions will be ONLY for first image

        Otherwise:
            np.ndarray
                Peaks
            np.ndarray
                Heatmap
            np.ndarray
                Paf

        """
        img_into_model = self._check_image(image)
        # Take predictions
        if resize_to is None:
            # Take `H`, `W` from input image
            resize_to = img_into_model[0].shape[:2]
        self._postprocess_class.set_resize_to(resize_to)
        feed_dict = {self._input_data_tensors[0]: img_into_model}

        if using_estimate_alg:
            paf, indices, peaks = self._postprocess_class(feed_dict)

            return SkeletBuilder.get_humans_by_PIF(
                    peaks=peaks.astype(np.float32, copy=False),
                    indices=indices.astype(np.int32, copy=False),
                    paf_mat=paf
            )

        return self._postprocess_class.get_data_for_debug(feed_dict)

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

    def _check_image(self, some_image) -> np.ndarray:
        """
        This method check input image into model and prepare it for further purposes

        """
        if len(some_image.shape) != 3:
            raise ValueError("Input image into model have wrong number of dims.\n" +\
                             f"Dims in input image equal to : {len(some_image.shape)}\n" +\
                             "But must be equal to 3"
            )

        proper_image = np.stack([some_image] * self.get_batch_size(), axis=0)
        return proper_image

