import tensorflow as tf
from makiflow.core import MakiTrainer, TrainerBuilder
from abc import ABC

EPS = 1e-6


class PETrainer(MakiTrainer, ABC):
    PAF_SCALE = 'paf_scale'             # float
    HEATMAP_SCALE = 'heatmap_scale'     # float
    HEATMAP_WEIGHT = 'heatmap_weight'   # float
    PAF_WEIGHT = 'paf_weight'           # float
<<<<<<< HEAD
    RESIZE_TO = 'resize_to'             # tuple of size 2 with value: (H, W)
=======
    RESIZE_TO = 'resize_to'                   # list(H, W)
>>>>>>> origin/add_new_pafs

    PAF_LOSS = 'PAF_loss'
    HEATMAP_LOSS = 'Heatmap_loss'

    __IDENTITY = 1.0

    TRAINING_PAF = 'TRAINING_PAF'
    TRAINING_HEATMAP = 'TRAINING_HEATMAP'
    TRAINING_MASK = 'TRAINING_MASK'

    def to_dict(self):
        return {
            TrainerBuilder.TYPE: self.TYPE,
            TrainerBuilder.PARAMS: {
                PETrainer.PAF_SCALE: self._paf_scale,
                PETrainer.HEATMAP_SCALE: self._heatmap_scale,
                PETrainer.HEATMAP_WEIGHT: self._heatmap_weight,
                PETrainer.PAF_WEIGHT: self._paf_weight,
                PETrainer.RESIZE_TO: self._resize_to
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

<<<<<<< HEAD
        resize_to = params.get(PETrainer.RESIZE_TO)
        if resize_to is None:
=======
        resize = params.get(PETrainer.RESIZE_TO)
        if resize is None:
>>>>>>> origin/add_new_pafs
            print('`resize` parameter is not set.')

        self.set_resize(resize_to)

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

    # noinspection PyAttributeOutsideInit
    def set_resize(self, resize_to):
        """
        Controls whether to resize of the training pafs, heatmaps and masks.

        Parameters
        ----------
        resize : list
            List of (H, W), if equal to None, will be not used

        """
<<<<<<< HEAD
        self._resize_to = resize_to
=======
        self._resize_to = resize
>>>>>>> origin/add_new_pafs

    def _init(self):
        super()._init()
        self._paf_scale = 1.0
        self._heatmap_scale = 1.0
        self._heatmap_weight = None
        self._paf_weight = None
        self._resize_to = None

    def _setup_label_placeholders(self):
        model = super().get_model()

        inference_paf = model.get_paf_makitensors()[0]
        paf_shape = inference_paf.get_shape()

        inference_heatmap = model.get_heatmap_makitensors()[0]
        heatmap_shape = inference_heatmap.get_shape()

        mask_shape = heatmap_shape[:-1] + [1]
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
            ),
            PETrainer.TRAINING_MASK: tf.placeholder(
                dtype='float32',
                shape=mask_shape,
                name=PETrainer.TRAINING_MASK
            )
        }

    def get_train_paf(self):
        label_tensors = super().get_label_tensors()
        pafs = label_tensors[PETrainer.TRAINING_PAF]
        if self._resize_to is not None:
            old_shape = pafs.get_shape()
            dymanic_shape = tf.shape(pafs)
            # [batch, h, w, pairs, 2]
            pafs_shape = old_shape.as_list()
            for i in range(len(pafs_shape)):
                if pafs_shape[i] is None:
                    pafs_shape[i] = dymanic_shape[i]

            # [batch, h, w, pairs, 2] --> [batch, h, w, pairs * 2]
            pafs = tf.reshape(pafs, pafs_shape[:3] + [-1])
            pafs = self.__resize_training_tensor(pafs)

            # [batch, h, w, pairs * 2] --> [batch, h, w, pairs, 2]
            new_pafs_shape = pafs.get_shape().as_list()
            pafs_shape[1] = new_pafs_shape[1]
            pafs_shape[2] = new_pafs_shape[2]
            pafs = tf.reshape(pafs, pafs_shape)
            new_shape = pafs.get_shape()
            print(f"Resized heatmap from old_shape={old_shape} to new_shape={new_shape}")
        return pafs

    def get_train_heatmap(self):
        label_tensors = super().get_label_tensors()
        heatmap = label_tensors[PETrainer.TRAINING_HEATMAP]
        if self._resize_to is not None:
            old_shape = heatmap.get_shape()
            heatmap = self.__resize_training_tensor(heatmap)
            new_shape = heatmap.get_shape()
            print(f"Resized heatmap from old_shape={old_shape} to new_shape={new_shape}")
        return heatmap

    def get_train_mask(self):
        label_tensors = super().get_label_tensors()
        mask = label_tensors[PETrainer.TRAINING_MASK]
        if self._resize_to is not None:
            old_shape = mask.get_shape()
            mask = self.__resize_training_tensor(mask)
            new_shape = mask.get_shape()
            print(f"Resized mask from old_shape={old_shape} to new_shape={new_shape}")
        return mask

    def __resize_training_tensor(self, tensor):
        tensor = tf.image.resize_area(
            tensor,
            self._resize_to,
            align_corners=False,
            name='tensor_resize'
        )
        return tensor

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





