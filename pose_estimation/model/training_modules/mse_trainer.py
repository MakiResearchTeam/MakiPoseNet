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


class MSETrainer(PETrainer):
    TYPE = 'MSETrainer'

    __IDENTITY = 1.0

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

        for heatmap in super().get_heatmap_tensors():
            # Division by 2.0 makes it similar to tf.nn.l2_loss
            heatmap_loss = Loss.mse_loss(train_heatmap, heatmap, raw_tensor=True) * train_heatmap_mask / 2.0

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


TrainerBuilder.register_trainer(MSETrainer)
