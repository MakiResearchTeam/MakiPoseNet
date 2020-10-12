import tensorflow as tf
from ..main_modules import PoseEstimatorInterface
from makiflow.base import MakiTensor, Loss
from makiflow.models.common.utils import print_train_info, moving_average
from makiflow.models.common.utils import new_optimizer_used, loss_is_built
from tqdm import tqdm


class MSETrainer:
    __MSG_LOSS_IS_BUILT = 'The loss tensor is already built. The next call of the fit method ' + \
                          'will rebuild it.'

    TOTAL_LOSS = 'Total loss'
    PAF_LOSS = 'PAF loss'
    HEATMAP_LOSS = 'Heatmap loss'

    def __init__(self, model: PoseEstimatorInterface, training_paf: MakiTensor, training_heatmap: MakiTensor):
        """
        Performs training of the model using hard masks.

        Parameters
        ----------
        model : PoseEstimatorInterface
            Model instance to train.
        training_paf : MakiTensor
            MakiTensor for training pafs.
        training_heatmap : MakiTensor
            MakiTensor for training heatmaps.
        """
        self._model = model
        model.training_on()
        paf_makitensors = model.get_paf_makitensor()
        heatmap_makitensors = model.get_heatmap_makitensor()
        paf_names = [x.get_name() for x in paf_makitensors]
        heatmap_names = [x.get_name() for x in heatmap_makitensors]

        self._paf_tensors = [
            model.get_traingraph_tensor(name) for name in paf_names
        ]
        self._heatmap_tensors = [
            model.get_traingraph_tensor(name) for name in heatmap_names
        ]
        self._training_paf = training_paf.get_data_tensor()
        self._training_heatmap = training_heatmap.get_data_tensor()
        self._paf_scale = 1.0
        self._heatmap_scale = 1.0
        self._loss_is_built = False
        self._optimizer = None
        self._sess = model.get_session()
        assert self._sess is not None
        self._tb_writer = None
        # Counter for total number of training iterations.
        self._tb_counter = 0

    def set_tensorboard_writer(self, writer):
        """
        Creates logging file for the Tensorboard in the given `logdir_path` directory.
        Parameters
        ----------
        logdir_path : str
            Path to the log directory.
        """
        self._tb_writer = writer

    def close_tensorboard(self):
        """
        Closes the logging writer for the Tensorboard
        """
        self._tb_writer.close()

    def set_loss_scales(self, paf_scale, heatmap_scale):
        """
        The paf loss and the heatmap loss will be scaled by these coefficients.
        Parameters
        ----------
        paf_scale : float
            Scale for the paf loss.
        heatmap_scale : float
            Scale for the heatmap loss.
        """
        if self._loss_is_built:
            print(MSETrainer.__MSG_LOSS_IS_BUILT)
            self._loss_is_built = False

        self._paf_scale = paf_scale
        self._heatmap_scale = heatmap_scale

    def _build_loss(self):
        paf_losses = []
        heatmap_losses = []

        for paf in self._paf_tensors:
            paf_losses.append(
                Loss.mse_loss(self._training_paf, paf)
            )

        for heatmap in self._heatmap_tensors:
            heatmap_losses.append(
                Loss.mse_loss(self._training_heatmap, heatmap)
            )

        sum_pafs = tf.reduce_sum(paf_losses)
        sum_heatmaps = tf.reduce_sum(heatmap_losses)

        losses = sum_heatmaps * self._heatmap_scale + \
                 sum_pafs * self._paf_scale

        loss = losses

        self._paf_loss = sum_pafs
        self._heatmap_loss = sum_heatmaps


        #self._paf_loss = 0.0
        #for paf in self._paf_tensors:
        #    self._paf_loss += Loss.mse_loss(self._training_paf, paf)

        #self._heatmap_loss = 0.0
        #for heatmap in self._heatmap_tensors:
        #    self._heatmap_loss += Loss.mse_loss(self._training_heatmap, heatmap)

        #loss = self._paf_scale * self._paf_loss + \
        #       self._heatmap_scale * self._heatmap_loss
        self._total_loss = self._model.build_final_loss(loss)

        # For Tensorboard
        paf_loss_summary = tf.summary.scalar(MSETrainer.PAF_LOSS, self._paf_loss)
        heatmap_loss_summary = tf.summary.scalar(MSETrainer.HEATMAP_LOSS, self._heatmap_loss)
        total_loss_summary = tf.summary.scalar(MSETrainer.TOTAL_LOSS, self._total_loss)
        self._summary = tf.summary.merge([
            paf_loss_summary,
            heatmap_loss_summary,
            total_loss_summary
        ])

    def _minimize_loss(self, optimizer, global_step):
        if not self._loss_is_built:
            self._build_loss()
            self._loss_is_built = True
            loss_is_built()

        if self._optimizer != optimizer:
            self._optimizer = optimizer
            self._train_op = optimizer.minimize(
                self._total_loss, var_list=self._model.get_training_vars(), global_step=global_step
            )
            self._sess.run(tf.variables_initializer(optimizer.variables()))
            new_optimizer_used()

        return self._train_op

    def fit(self, optimizer, epochs=1, iter=10, print_period=None, global_step=None):
        """
        Performs fitting of the model.

        Parameters
        ----------
        optimizer : TensorFlow optimizer
            Model uses TensorFlow optimizers in order train itself.
        epochs : int
            Number of epochs to run.
        iter : int
            Number of training iterations per update.
        print_period : int
            Every `print_period` training iterations the training info will be displayed.
        global_step
            Please refer to TensorFlow documentation about global step for more info.
        Returns
        -------
        dict
            Dictionary with information about: total loss, paf loss, heatmap loss.
        """
        train_op = self._minimize_loss(optimizer, global_step)

        if print_period is None:
            print_period = iter

        total_losses = []
        heatmap_losses = []
        paf_losses = []
        
        for i in range(epochs):
            it = tqdm(range(iter))
            total_loss = 0.0
            paf_loss = 0.0
            heatmap_loss = 0.0

            for j in it:
                b_total_loss, b_paf_loss, b_heatmap_loss, summary, _ = self._sess.run(
                    [self._total_loss, self._paf_loss, self._heatmap_loss, self._summary, train_op]
                )
                total_loss = moving_average(total_loss, b_total_loss, j)
                paf_loss = moving_average(paf_loss, b_paf_loss, j)
                heatmap_loss = moving_average(heatmap_loss, b_heatmap_loss, j)

                self._tb_counter += 1
                if (j + 1) % print_period == 0:
                    print_train_info(
                        i,
                        (MSETrainer.TOTAL_LOSS, total_loss),
                        (MSETrainer.PAF_LOSS, paf_loss),
                        (MSETrainer.HEATMAP_LOSS, heatmap_loss)
                    )
                    if self._tb_writer is not None:
                        self._tb_writer.add_summary(summary, self._tb_counter)
                        #self._tb_writer.flush()
                        #print('OK!')

            total_losses.append(total_loss)
            heatmap_losses.append(heatmap_loss)
            paf_losses.append(paf_loss)
        return {
            MSETrainer.TOTAL_LOSS: total_losses,
            MSETrainer.HEATMAP_LOSS: heatmap_losses,
            MSETrainer.PAF_LOSS: paf_losses
        }
