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
            upsample_size: list,
            **kwargs):
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
        paf_label_layer : MakiLayer
        heatmap_label_layer : MakiLayer

        Returns
        -------
        paf: TODO wrapper
        heatmap : TODO wrapper

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

    def __label_correction_paf(self, t_paf: tf.Tensor, l_paf: tf.Tensor):
        t_norm = tf.nn.l2_normalize(t_paf)
        l_norm = tf.nn.l2_normalize(l_paf)

        corrected_paf = tf.where(
            tf.greater(l_norm, t_norm),
            x=l_paf,     # true
            y=t_paf      # false
        )

        return corrected_paf

    def __label_correction_heatmap(self, t_heatmap: tf.Tensor, l_heatmap: tf.Tensor):
        stacked_heatmap = tf.stack([t_heatmap, l_heatmap], axis=-1)
        # apply max
        corrected_heatmap = tf.reduce_max(
            stacked_heatmap, axis=-1
        )
        return corrected_heatmap

