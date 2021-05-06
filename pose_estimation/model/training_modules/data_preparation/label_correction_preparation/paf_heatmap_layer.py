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

from makiflow.core import MakiTensor
from makiflow.tools.tf_tools import load_graph_def
import tensorflow as tf
from .maki_layer_wrapper import MakiLayerWrapper


class PHLabelCorrectionLayer:
    """
    Paf and Heatmap Label correction layer
    Most of ideas taken from:
    https://arxiv.org/pdf/1811.03331.pdf

    """

    def __init__(
            self,
            model_pb_path: str,
            input_layer_name: str,
            paf_layer_name: str,
            heatmap_layer_name: str,
            upsample_size_tensor_name: str,
            upsample_size: list):
        """

        Parameters
        ----------
        model_pb_path : str
            Path to model in pb format.
            Example: '/home/user/model.pb'
        input_layer_name : str
            Name of the input layer.
            Example: 'Input:0'
            NOTICE! :0 - is very important, because without it - it will be operation name, NOT tensor itself
        paf_layer_name : str
            Name of the paf layer.
            Example: 'paf_stuf:0'
            NOTICE! :0 - is very important, because without it - it will be operation name, NOT tensor itself
        heatmap_layer_name : str
            Name of the heatmap layer.
            Example: 'heatmap_stuf:0'
            NOTICE! :0 - is very important, because without it - it will be operation name, NOT tensor itself
        upsample_size_tensor_name : str
            Name of the upsample_size layer.
            Example: 'upsample_size:0'
            NOTICE! :0 - is very important, because without it - it will be operation name, NOT tensor itself
        upsample_size : list
            Value that will be set into `upsample_size` tensor

        """
        self._model_pb_path = model_pb_path
        self._input_layer_name = input_layer_name
        self._paf_layer_name = paf_layer_name
        self._heatmap_layer_name = heatmap_layer_name
        self._upsample_size_tensor_name = upsample_size_tensor_name
        self._upsample_size = tf.constant(upsample_size, dtype=tf.int32)

    def compile(self, input_image: tf.Tensor, paf_label_layer: MakiTensor, heatmap_label_layer: MakiTensor):
        """
        Apply label correction for label heatmap and paf tensors by taken output from teacher

        Parameters
        ----------
        input_image : tf.Tensor
        paf_label_layer : MakiTensor
        heatmap_label_layer : MakiTensor

        Returns
        -------
        paf : MakiLayerWrapper
            Final paf layer with correction
        heatmap : MakiLayerWrapper
            Final heatmap layer with correction

        """
        teacher_paf_tensor, teacher_heatmap_tensor = self.__init_teacher(input_image)
        paf_corrected = self.__label_correction_paf(
            t_paf=teacher_paf_tensor,
            l_paf=paf_label_layer.get_data_tensor()
        )

        heatmap_corrected = self.__label_correction_heatmap(
            t_heatmap=teacher_heatmap_tensor,
            l_heatmap=heatmap_label_layer.get_data_tensor()
        )

        paf_mf = MakiLayerWrapper(paf_corrected)
        heatmap_mf = MakiLayerWrapper(heatmap_corrected)

        return paf_mf, heatmap_mf

    def __init_teacher(self, input_tensor: tf.Tensor):
        """
        Load model and weights

        """
        self.__graph_def = load_graph_def(self._model_pb_path)

        teacher_paf_tensor, teacher_heatmap_tensor = tf.import_graph_def(
            self.__graph_def,
            input_map={
                self._input_layer_name: input_tensor,
                self._upsample_size_tensor_name: self._upsample_size
            },
            return_elements=[
                self._paf_layer_name,
                self._heatmap_layer_name
            ]
        )
        teacher_paf_tensor = tf.convert_to_tensor(teacher_paf_tensor, dtype=tf.float32)
        teacher_heatmap_tensor = tf.convert_to_tensor(teacher_heatmap_tensor, dtype=tf.float32)

        return teacher_paf_tensor, teacher_heatmap_tensor

    def __label_correction_paf(self, t_paf: tf.Tensor, l_paf: tf.Tensor) -> tf.Tensor:
        """
        Correct paf
        """
        assert len(t_paf.get_shape()) == 5 and len(l_paf.get_shape()) == 5, \
            f'Expected paf tensors to dimensionality of 5, but received dim(l_paf)={len(l_paf.get_shape())}, ' \
            f'dim(t_paf)={len(t_paf.get_shape())}.'

        l2_norm = lambda t: tf.reduce_sum(t*t, axis=-1)
        l_norm = l2_norm(l_paf)  # label
        t_norm = l2_norm(t_paf)  # teacher
        # Apply correction
        # We need to stack the boolean tensor twice so that it has the same shape as the paf tensors
        condition = tf.stack([tf.greater(l_norm, t_norm)]*2, axis=-1)
        corrected_paf = tf.where(
            condition,
            x=l_paf,     # true
            y=t_paf      # false
        )
        return corrected_paf

    def __label_correction_heatmap(self, t_heatmap: tf.Tensor, l_heatmap: tf.Tensor) -> tf.Tensor:
        """
        Correct heatmap

        """
        stacked_heatmap = tf.stack([t_heatmap, l_heatmap], axis=-1)
        # Apply max
        # So, we take best from teacher and label
        corrected_heatmap = tf.reduce_max(
            stacked_heatmap, axis=-1
        )
        return corrected_heatmap

