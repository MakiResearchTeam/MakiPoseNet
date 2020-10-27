import tensorflow as tf
from makiflow.core import MakiTrainer, TrainerBuilder
from abc import ABC

EPS = 1e-6


class PETrainer(MakiTrainer, ABC):
    PAF_SCALE = 'paf_scale'
    HEATMAP_SCALE = 'heatmap_scale'
    HEATMAP_WEIGHT = 'heatmap_weight'
    PAF_WEIGHT = 'paf_weight'

    PAF_LOSS = 'PAF_loss'
    HEATMAP_LOSS = 'Heatmap_loss'

    __IDENTITY = 1.0

    TRAINING_PAF = 'TRAINING_PAF'
    TRAINING_HEATMAP = 'TRAINING_HEATMAP'

    def to_dict(self):
        return {
            TrainerBuilder.TYPE: self.TYPE,
            TrainerBuilder.PARAMS: {
                PETrainer.PAF_SCALE: self._paf_scale,
                PETrainer.HEATMAP_SCALE: self._heatmap_scale,
                PETrainer.HEATMAP_WEIGHT: self._heatmap_weight,
                PETrainer.PAF_WEIGHT: self._paf_weight
            }
        }

    def setup_params(self, params):
        heatmap_scale = params[PETrainer.HEATMAP_SCALE]
        paf_scale = params[PETrainer.PAF_SCALE]

        heatmap_weight = params[PETrainer.HEATMAP_WEIGHT]
        paf_weight = params[PETrainer.PAF_WEIGHT]

        self.set_loss_scales(
            heatmap_scale=heatmap_scale,
            paf_scale=paf_scale,
            heatmap_weight=heatmap_weight,
            paf_weight=paf_weight
        )

    def _init(self):
        super()._init()
        self._paf_scale = None
        self._heatmap_scale = None
        self._heatmap_weight = None
        self._paf_weight = None

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

    def get_heatmap_tensors(self):
        inference_heatmaps = super().get_model().get_heatmap_makitensors()
        traingraph_heatmaps = []
        for heatmap in inference_heatmaps:
            name = heatmap.get_name()
            tensor = super().get_traingraph_tensor(name)
            traingraph_heatmaps.append(tensor)

        return traingraph_heatmaps

    def get_paf_tensors(self):
        inference_pafs = super().get_model().get_paf_makitensors()
        traingraph_pafs = []
        for paf in inference_pafs:
            name = paf.get_name()
            tensor = super().get_traingraph_tensor(name)
            traingraph_pafs.append(tensor)

        return traingraph_pafs

    def get_label_feed_dict_config(self):
        label_tensors = super().get_label_tensors()
        return {
            label_tensors[PETrainer.TRAINING_PAF]: 0,
            label_tensors[PETrainer.TRAINING_HEATMAP]: 1
        }

    # noinspection PyAttributeOutsideInit
    def set_loss_scales(
            self,
            paf_scale=None,
            heatmap_scale=None,
            paf_weight=None,
            heatmap_weight=None):
        """
        The paf loss and the heatmap loss will be scaled by these coefficients.
        Parameters
        ----------
        paf_scale : float
            Scale for the paf loss.
        heatmap_scale : float
            Scale for the heatmap loss.
        paf_weight : float
            Scale for weights mask for every given paf into this model,
            If equal to None, weights masks will be not used
        heatmap_weight : float
            Scale for weights mask for every given heatmap into this model,
            If equal to None, weights masks will be not used
        """
        if paf_scale is not None:
            self._paf_scale = paf_scale

        if heatmap_scale is not None:
            self._heatmap_scale = heatmap_scale

        if heatmap_weight is not None:
            self._heatmap_weight = heatmap_weight

        if paf_weight is not None:
            self._paf_weight = paf_weight



