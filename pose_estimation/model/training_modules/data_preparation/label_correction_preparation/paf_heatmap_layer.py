from makiflow.core import MakiLayer
import tensorflow as tf
from .heatmap_wrapper import HeatmapWrapperLayer
from .paf_wrapper import PafWrapperLayer


class PHLabelCorrectionLayer:
    """
    Paf and Heatmap Label correction layer
    Most of ideas taken from:
    https://arxiv.org/pdf/1811.03331.pdf

    """

    def __init__(
            self,
            model_teacher_path: str,
            model_weights_path: str,
            **kwargs):
        self.__init_teacher(model_teacher_path=model_teacher_path, model_weights_path=model_weights_path)

    def __init_teacher(self, model_teacher_path: str, model_weights_path: str):
        """
        Load model and weights

        """
        pass

    def compile(self, input_image: tf.Tensor, paf_label_layer: MakiLayer, heatmap_label_layer: MakiLayer):
        """
        Apply label correction for label heatmap and paf tensors by taken output from teacher

        Parameters
        ----------
        input_image : tf.Tensor
        paf_layer : MakiLayer
        heatmap_layer : MakiLayer

        Returns
        -------
        paf: TODO wrapper
        heatmap : TODO wrapper

        """
        pass

