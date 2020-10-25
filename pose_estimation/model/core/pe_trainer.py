import tensorflow as tf
from makiflow.core import MakiTrainer
from abc import ABC

EPS = 1e-6


class PETrainer(MakiTrainer, ABC):
    def _setup_for_training(self):
        super()._setup_for_training()
        self._paf_scale = None
        self._heatmap_scale = None
        self._heatmap_single_scale = None
        self._paf_single_scale = None

    def _setup_label_placeholders(self):
        model = super().get_model()

        inference_paf = model.get_paf_makitensors()
        paf_shape = inference_paf.get_shape()

        inference_heatmap = model.get_heatmap_makitensors()
        heatmap_shape = inference_heatmap.get_shape()
        return {
            PETrainer.TRAINING_PAF: tf.placeholder(
                dtype='float32',
                shape=paf_shape,
                name=PETrainer.TRAINING_PAF
            ),
            PETrainer.TRAINING_HEATMAP: tf.placeholder(
                dtype='float32',
                shape=heatmap_shape,
                name=PETrainer.TRAINING_HEATMAP
            )
        }

    def get_train_paf(self):
        label_tensors = super().get_label_tensors()
        return label_tensors[PETrainer.TRAINING_PAF]

    def get_train_heatmap(self):
        label_tensors = super().get_label_tensors()
        return label_tensors[PETrainer.TRAINING_HEATMAP]

    def _init(self):
        super()._init()
        inference_pafs = super().get_model().get_paf_makitensors()
        self._traingraph_pafs = []
        for paf in inference_pafs:
            name = paf.get_name()
            tensor = super().get_traingraph_tensor(name)
            self._traingraph_pafs.append(tensor)

        inference_heatmaps = super().get_model().get_heatmap_makitensors()
        self._traingraph_heatmaps = []
        for heatmap in inference_heatmaps:
            name = heatmap.get_name()
            tensor = super().get_traingraph_tensor(name)
            self._traingraph_heatmaps.append(tensor)

    def get_heatmap_tensors(self):
        return self._traingraph_heatmaps

    def get_paf_tensors(self):
        return self._traingraph_pafs

    def get_label_feed_dict_config(self):
        label_tensors = super().get_label_tensors()
        return {
            label_tensors[PETrainer.TRAINING_PAF]: 0,
            label_tensors[PETrainer.TRAINING_HEATMAP]: 1
        }

    MSE_LOSS = 'mse_loss'

    PAF_SCALE = 'paf_scale'
    HEATMAP_SCALE = 'heatmap_scale'
    HEATMAP_SINGLE_SCALE = 'heatmap_single_scale'
    PAF_SINGLE_SCALE = 'paf_single_scale'

    PAF_LOSS = 'PAF_loss'
    HEATMAP_LOSS = 'Heatmap_loss'

    __IDENTITY = 1.0

    TRAINING_PAF = 'TRAINING_PAF'
    TRAINING_HEATMAP = 'TRAINING_HEATMAP'

    def set_loss_scales(
            self,
            paf_scale=None,
            heatmap_scale=None,
            paf_single_scale=None,
            heatmap_single_scale=None):
        """
        The paf loss and the heatmap loss will be scaled by these coefficients.
        Parameters
        ----------
        paf_scale : float
            Scale for the paf loss.
        heatmap_scale : float
            Scale for the heatmap loss.
        paf_single_scale : float
            Scale for weights mask for every given paf into this model,
            If equal to None, weights masks will be not used
        heatmap_single_scale : float
            Scale for weights mask for every given heatmap into this model,
            If equal to None, weights masks will be not used
        """
        if paf_scale is not None:
            self._paf_scale = paf_scale

        if heatmap_scale is not None:
            self._heatmap_scale = heatmap_scale

        if heatmap_single_scale is not None:
            self._heatmap_single_scale = heatmap_single_scale

        if paf_single_scale is not None:
            self._paf_single_scale = paf_single_scale



