import tensorflow as tf
from ..core import PETrainer
from makiflow.core import Loss


EPS = 1e-6


class MSETrainer(PETrainer):
    __IDENTITY = 1.0

    def _build_loss(self):
        train_paf = super().get_train_paf()
        train_heatmap = super().get_train_heatmap()

        paf_losses = []
        heatmap_losses = []
        for paf in super().get_paf_tensors():
            single_paf_loss = Loss.mse_loss(train_paf, paf, raw_tensor=True)

            if self._paf_single_scale is not None:
                abs_training_paf = tf.abs(train_paf)

                multiplied_single_loss = tf.cast(
                    tf.math.greater(abs_training_paf, EPS),
                    dtype=tf.float32
                )

                scale_for_loss = multiplied_single_loss * self._paf_single_scale + self.__IDENTITY

                single_paf_loss = single_paf_loss * scale_for_loss

            paf_losses.append(
                tf.reduce_mean(single_paf_loss)
            )

        for heatmap in super().get_heatmap_tensors():
            single_heatmap_loss = Loss.mse_loss(train_heatmap, heatmap, raw_tensor=True)

            if self._heatmap_single_scale is not None:
                # Create mask for scaling loss
                # Add 1.0 for saving values that are equal to 0 (approximately equal to 0)
                mask_heatmap_single = train_heatmap * self._heatmap_single_scale + self.__IDENTITY

                single_heatmap_loss = single_heatmap_loss * mask_heatmap_single

            heatmap_losses.append(
                tf.reduce_mean(single_heatmap_loss)
            )

        self._paf_loss = tf.reduce_sum(paf_losses)
        self._heatmap_loss = tf.reduce_sum(heatmap_losses)

        loss = self._heatmap_loss * self._heatmap_scale + \
               self._paf_loss * self._paf_scale

        # For Tensorboard
        super().track_loss(self._paf_loss, PETrainer.PAF_LOSS)
        super().track_loss(self._heatmap_loss, PETrainer.HEATMAP_LOSS)

        return loss
