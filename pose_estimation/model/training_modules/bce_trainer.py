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

import tensorflow as tf
from ..core import PETrainer
from makiflow.core import Loss, TrainerBuilder

EPS = 1e-6


class BCETrainer(PETrainer):
    TYPE = 'BCETrainer'

    __IDENTITY = 1.0
    NULLIFY_LOSS = 'nullify_loss'
    LABEL_SMOOTHING = 'label_smoothing'

    def to_dict(self):
        trainer_dict = super(BCETrainer, self).to_dict()
        trainer_dict[PETrainer.PARAMS][BCETrainer.NULLIFY_LOSS] = self._is_nullify_absent_labels
        return trainer_dict

    def set_params(self, params):
        super(BCETrainer, self).set_params(params)

        nullify = params.get(BCETrainer.NULLIFY_LOSS)
        if nullify:
            self.set_nullify_absent_labels()

        label_smoothing = params.get(BCETrainer.LABEL_SMOOTHING)
        if label_smoothing is not None:
            self.set_label_smoothing(label_smoothing)

    def set_nullify_absent_labels(self):
        """
        If a heatmap label mask equals zero everywhere, the loss will be multiplies by zero.
        """
        self._is_nullify_absent_labels = True

    def set_label_smoothing(self, smoothing_factor):
        """
        Sets a label smoothing to be used during training.
        0 - no label smoothing to be used. By default no label smoothing is used.

        Parameters
        ----------
        smoothing_factor : float
        """
        assert 0 <= smoothing_factor <= 1, 'Smoothing factor must be in range [0, 1], but' \
                                                                f'received {smoothing_factor}'
        self._label_smoothing = smoothing_factor

    def _init(self):
        super()._init()
        self._is_nullify_absent_labels = False
        self._label_smoothing = 0

    def _build_loss(self):
        train_paf, train_paf_mask = super().get_train_paf()
        train_heatmap, train_heatmap_mask = super().get_train_heatmap()
        train_mask = super().get_train_mask()

        if train_paf_mask is not None:
            train_paf_mask = train_paf_mask * tf.expand_dims(train_mask, axis=-1)
        else:
            train_paf_mask = tf.expand_dims(train_mask, axis=-1)

        if train_heatmap_mask is not None:
            train_heatmap_mask = train_heatmap_mask * train_mask
        else:
            train_heatmap_mask = train_mask

        paf_losses = []
        heatmap_losses = []
        # --- PAF LOSS
        for paf in super().get_paf_tensors():
            # Division by 2.0 makes it similar to tf.nn.l2_loss
            paf_loss = Loss.mse_loss(train_paf, paf, raw_tensor=True) * train_paf_mask / 2.0

            if self._paf_weight is not None:
                abs_training_paf = tf.abs(train_paf)

                mask = tf.cast(
                    tf.math.greater(abs_training_paf, EPS),
                    dtype=tf.float32
                )

                weights_mask = mask * self._paf_weight + self.__IDENTITY

                paf_loss = paf_loss * weights_mask

            paf_losses.append(
                tf.reduce_sum(paf_loss)
            )
        # --- HEATMAP LOSS
        for heatmap in super().get_heatmap_tensors():
            # We need to expand dims first, because tf.keras.losses.binary_crossentropy averages
            # the loss along the last dimension: [BS, H, W, C] -> [BS, H, W]
            # [BS, H, W, C, 1]
            train_heatmap_expanded = tf.expand_dims(train_heatmap, axis=-1)
            heatmap_expanded = tf.expand_dims(heatmap, axis=-1)
            heatmap_loss = tf.keras.losses.binary_crossentropy(
                y_true=train_heatmap_expanded, y_pred=heatmap_expanded, label_smoothing=self._label_smoothing
            ) * train_heatmap_mask / 2.0
            # heatmap_loss - [BS, H, W, C]

            if self._is_nullify_absent_labels:
                # [bs, 1, 1, C]
                label_sum = tf.reduce_sum(train_heatmap * train_mask, axis=[1, 2], keepdims=True)
                scale_factor = label_sum / (label_sum + 1e-5)
                heatmap_loss = heatmap_loss * scale_factor

            if self._heatmap_weight is not None:
                # Create mask for scaling loss
                # Add 1.0 for saving values that are equal to 0 (approximately equal to 0)
                weight_mask = train_heatmap * self._heatmap_weight + self.__IDENTITY
                heatmap_loss = heatmap_loss * weight_mask

            heatmap_losses.append(
                tf.reduce_sum(heatmap_loss)
            )

        # The original repo takes mean over the sums of the losses
        self._paf_loss = tf.reduce_mean(paf_losses)
        self._heatmap_loss = tf.reduce_mean(heatmap_losses)

        loss = self._heatmap_loss * self._heatmap_scale + \
               self._paf_loss * self._paf_scale

        # For Tensorboard
        super().track_loss(self._paf_loss, PETrainer.PAF_LOSS)
        super().track_loss(self._heatmap_loss, PETrainer.HEATMAP_LOSS)

        return loss


TrainerBuilder.register_trainer(BCETrainer)
