import tensorflow as tf
from ..core import PETrainer
from makiflow.core import Loss, TrainerBuilder

EPS = 1e-6


class MSETrainer(PETrainer):
    TYPE = 'MSETrainer'

    __IDENTITY = 1.0

    def _build_loss(self):
        train_paf = super().get_train_paf()
        train_heatmap = super().get_train_heatmap()
        train_mask = super().get_train_mask()

        paf_losses = []
        heatmap_losses = []
        for paf in super().get_paf_tensors():
            paf_loss = Loss.mse_loss(train_paf, paf, raw_tensor=True)

            if self._paf_weight is not None:
                abs_training_paf = tf.abs(train_paf)

                mask = tf.cast(
                    tf.math.greater(abs_training_paf, EPS),
                    dtype=tf.float32
                )

                weights_mask = mask * self._paf_weight + self.__IDENTITY

                paf_loss = paf_loss * weights_mask

            paf_losses.append(
                tf.reduce_mean(paf_loss)
            )

        for heatmap in super().get_heatmap_tensors():
            heatmap_loss = Loss.mse_loss(train_heatmap, heatmap, raw_tensor=True)

            if self._heatmap_weight is not None:
                # Create mask for scaling loss
                # Add 1.0 for saving values that are equal to 0 (approximately equal to 0)
                weight_mask = train_heatmap * self._heatmap_weight + self.__IDENTITY

                heatmap_loss = heatmap_loss * weight_mask

            heatmap_losses.append(
                tf.reduce_mean(heatmap_loss)
            )

        self._paf_loss = tf.reduce_sum(paf_losses)
        self._heatmap_loss = tf.reduce_sum(heatmap_losses)

        loss = self._heatmap_loss * self._heatmap_scale + \
               self._paf_loss * self._paf_scale

        # For Tensorboard
        super().track_loss(self._paf_loss, PETrainer.PAF_LOSS)
        super().track_loss(self._heatmap_loss, PETrainer.HEATMAP_LOSS)

        return loss


TrainerBuilder.register_trainer(MSETrainer)
