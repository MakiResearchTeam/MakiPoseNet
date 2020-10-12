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

        self._setup_tensorboard_vars()

    def _setup_tensorboard_vars(self):
        self._tb_is_setup = False
        self._tb_writer = None
        # Counter for total number of training iterations.
        self._tb_counter = 0
        self._tb_summaries = []

        self._grads_and_vars = None
        self._gradients = None

        self._model_layers = self._model.get_layers()
        # Required for plotting histograms
        self._layer_weights = {}
        for layer_name in self._model_layers:
            self._layer_weights[layer_name] = self._model_layers[layer_name].get_params()

        self._layers_weights_histograms = []
        self._layers_grads_histograms = []

    def add_summary(self, summary):
        self._tb_summaries.append(summary)

    def set_tensorboard_writer(self, writer):
        """
        Creates logging file for the Tensorboard in the given `logdir_path` directory.
        Parameters
        ----------
        writer : tf.FileWriter
            Path to the log directory.
        """
        self._tb_writer = writer

    def add_layers_weights_histograms(self, layer_names):
        # noinspection PyAttributeOutsideInit
        self._layers_weights_histograms = layer_names

    def add_layers_grads_histograms(self, layer_names):
        # noinspection PyAttributeOutsideInit
        self._layers_grads_histograms = layer_names

    def close_tensorboard(self):
        """
        Closes the logging writer for the Tensorboard
        """
        self._tb_writer.close()

    # Currently not used
    def _find_layer_owner_name(self, var):
        """
        Searches for a layer-owner name of the tf.Variable `var`.
        Parameters
        ----------
        var : tf.Variable
            Variable which owner's name to find.

        Returns
        -------
        str
            Name of the layer-owner.
        """
        for layer_name in self._layer_weights:
            if var is self._layer_weights[layer_name]:
                return layer_name
        raise ValueError(f'Could not find a layer-owner of the var = {var}')

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

    def _minimize_loss(self, optimizer, global_step):
        if not self._loss_is_built:
            self._build_loss()
            self._loss_is_built = True
            loss_is_built()

        if self._optimizer != optimizer:
            self._create_train_op(optimizer, global_step)

        if not self._tb_is_setup:
            self._setup_tensorboard()

        return self._train_op

    def _build_loss(self):
        self._paf_loss = 0.0
        for paf in self._paf_tensors:
            self._paf_loss += Loss.mse_loss(self._training_paf, paf)

        self._heatmap_loss = 0.0
        for heatmap in self._heatmap_tensors:
            self._heatmap_loss += Loss.mse_loss(self._training_heatmap, heatmap)

        loss = self._paf_scale * self._paf_loss + \
               self._heatmap_scale * self._heatmap_loss
        self._total_loss = self._model.build_final_loss(loss)

        # For Tensorboard
        paf_loss_summary = tf.summary.scalar(MSETrainer.PAF_LOSS, self._paf_loss)
        self.add_summary(paf_loss_summary)

        heatmap_loss_summary = tf.summary.scalar(MSETrainer.HEATMAP_LOSS, self._heatmap_loss)
        self.add_summary(heatmap_loss_summary)

        total_loss_summary = tf.summary.scalar(MSETrainer.TOTAL_LOSS, self._total_loss)
        self.add_summary(total_loss_summary)

    def _create_train_op(self, optimizer, global_step):
        self._optimizer = optimizer

        if self._grads_and_vars is None:
            training_vars = self._model.get_training_vars()
            # Returns list of tuples: [ (grad, var) ]
            self._grads_and_vars = optimizer.compute_gradients(self._total_loss, training_vars)
            vars_and_grads = [(var, grad) for grad, var in self._grads_and_vars]
            # Collect mapping from the variable to its grad for tensorboard
            self._var2grad = dict(vars_and_grads)

        self._train_op = optimizer.apply_gradients(
            grads_and_vars=self._grads_and_vars, global_step=global_step
        )

        self._sess.run(tf.variables_initializer(optimizer.variables()))
        new_optimizer_used()

    def _setup_tensorboard(self):
        assert len(self._tb_summaries) != 0, 'No summaries found.'
        self._total_summary = tf.summary.merge(self._tb_summaries)

        # Collect all weights histograms
        for layer_name in self._layers_weights_histograms:
            with tf.name_scope(f'weight/{layer_name}'):
                for weight in self._model.get_layer(layer_name):
                    self.add_summary(tf.summary.histogram(name=weight.name, values=weight))

        # Collect all grads histograms
        for layer_name in self._layers_weights_histograms:
            with tf.name_scope(f'grad/{layer_name}'):
                for weight in self._model.get_layer(layer_name):
                    grad = self._var2grad.get(weight)
                    if grad is None:
                        print(f'Did not find gradient for layer={layer_name}, var={weight.name}')
                        continue
                    self.add_summary(tf.summary.histogram(name=weight.name, values=weight))

        self._total_summary = tf.summary.merge(self._tb_summaries)
        self._tb_is_setup = True

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
            total_loss = 0
            paf_loss = 0
            heatmap_loss = 0
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
