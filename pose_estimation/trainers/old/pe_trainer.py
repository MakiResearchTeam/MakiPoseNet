
from .pe_trainer_interface import (PETrainerInterface, ExpField,
                                   SubExpField, LossType )

import json
import os
import tensorflow as tf
import numpy as np
import traceback
import cv2
import glob
import copy
from scipy.stats import skew, kurtosis
import pandas as pd

from makiflow.trainers.utils.optimizer_builder import OptimizerBuilder
from makiflow.tools.test_visualizer import TestVisualizer


class PETrainer:

    def __init__(
            self,
            model_creation_function,
            exp_params: str,
            path_to_save: str):
        """
        Initialize Render trainer.

        Parameters
        ----------
            model_creation_function : function
                Function that create model. API of the function is next:
                    model_creation_function(use_gen, batch_size, sess)
                where:
                    use_gen : bool
                        If true, then function return model with generator (usually this is for training)
                    batch_size : int
                        Batch size of the model
                    sess : tensorflow object
                        Current session for special usage.
            exp_params : str
                Path to json file with parameters for training.
            path_to_save : str
                Path for experiments folder. If its does not exist, it will be created.

        """
        self._exp_params = exp_params
        self._model_creation_function = model_creation_function

        if type(exp_params) is str:
            self._exp_params = self._load_exp_params_from_json(exp_params)
        else:
            raise TypeError("`exp_params` must be path to JSON file with parameters for training")

        self._path_to_save = path_to_save
        self._sess = None
        self.generator = None

    def _load_exp_params_from_json(self, json_path):
        """
        Read parameters for experiments from json file

        Parameters
        ----------
        json_path : str
            Path to json file with parameters

        Returns
        -------
        dict
            Dictionary with parameters for experiments

        """
        with open(json_path) as json_file:
            json_value = json_file.read()
            exp_params = json.loads(json_value)

        return exp_params

    def start_experiments(self):
        """
        Starts all experiments.

        """
        for experiment in self._exp_params[ExpField.EXPERIMENTS]:
            self._start_exp(experiment)

    def _start_exp(self, experiment):
        """
        Setting up parameters for experiment and starting it

        """
        self._create_experiment_folder(experiment[ExpField.NAME])

        exp_params = {
            ExpField.NAME: experiment[ExpField.NAME],
            ExpField.PRETRAINED_LAYERS: experiment[ExpField.PRETRAINED_LAYERS],
            ExpField.WEIGHTS: experiment[ExpField.WEIGHTS],
            ExpField.UNTRAINABLE_LAYERS: experiment[ExpField.UNTRAINABLE_LAYERS],
            ExpField.EPOCHS: experiment[ExpField.EPOCHS],
            ExpField.TEST_PERIOD: experiment[ExpField.TEST_PERIOD],
            ExpField.LOSS_TYPE: experiment[ExpField.LOSS_TYPE],
            ExpField.METRIC_TYPE: experiment[ExpField.METRIC_TYPE],
            ExpField.L1_REG: experiment[ExpField.L1_REG],
            ExpField.L1_REG_LAYERS: experiment[ExpField.L1_REG_LAYERS],
            ExpField.L2_REG: experiment[ExpField.L2_REG],
            ExpField.L2_REG_LAYERS: experiment[ExpField.L2_REG_LAYERS],
            ExpField.PATH_TEST_IMAGE: experiment[ExpField.PATH_TEST_IMAGE],
            ExpField.BATCH_SIZE: experiment[ExpField.BATCH_SIZE],
            ExpField.ITERATIONS: experiment[ExpField.ITERATIONS],
        }

        for i, opt_info in enumerate(experiment[ExpField.OPTIMIZERS]):
            exp_params[SubExpField.OPT_INFO] = opt_info
            self._run_experiment(exp_params, i)

    def _run_experiment(self, exp_params, number_of_experiment):
        self._restore_model(exp_params)

        loss_type = exp_params[ExpField.LOSS_TYPE]
        opt_info = exp_params[SubExpField.OPT_INFO]
        epochs = exp_params[ExpField.EPOCHS]
        iterations = exp_params[ExpField.ITERATIONS]
        test_period = exp_params[ExpField.TEST_PERIOD]
        optimizer, global_step = OptimizerBuilder.build_optimizer(opt_info)
        if global_step is not None:
            self._sess.run(tf.variables_initializer([global_step]))

        # Catch InterruptException
        try:
            for i in range(epochs):
                if loss_type == LossType.ABS_LOSS:
                    sub_train_info = self._model.gen_fit_abs(optimizer=optimizer, epochs=1,
                                                             iterations=iterations, global_step=global_step)
                    loss_value = sub_train_info[ABS_LOSS][0]
                elif loss_type == LossType.MSE_LOSS:
                    sub_train_info = self._model.gen_fit_mse(optimizer=optimizer, epochs=1,
                                                             iterations=iterations, global_step=global_step)
                    loss_value = sub_train_info[MSE_LOSS][0]
                else:
                    raise ValueError(f'Unknown loss type {loss_type}!')

                self.loss_list += [loss_value]

                # For generators we should save weights and then load them into new model to perform test
                if i % test_period == 0:
                    save_path = f'{self._exp_folder}/{number_of_experiment}_exp/epoch_{i}'
                    os.makedirs(
                        save_path, exist_ok=True
                    )
                    self._model.save_weights(os.path.join(save_path, 'weights.ckpt'))

                    self._perform_testing(exp_params, save_path, os.path.join(save_path, 'weights.ckpt'))
                print('Epochs:', i)
        except KeyboardInterrupt as ex:
            traceback.print_exc()
            print("SAVING GAINED DATA")
        finally:
            # ALWAYS DO LAST SAVE
            save_path = f'{self._exp_folder}/{number_of_experiment}_exp'
            os.makedirs(
                save_path + '/last_weights', exist_ok=True
            )
            self._model.save_weights(f'{save_path}/last_weights/weights.ckpt')
            self._perform_testing(exp_params, save_path + '/last_weights/', save_path + '/last_weights/weights.ckpt')
            print('Test finished.')

            # Close the session since Generator yields unexpected behaviour otherwise.
            # Process doesn't stop until KeyboardInterruptExceptions occurs.
            # It also yield the following warning message:
            # 'Error occurred when finalizing GeneratorDataset iterator:
            # Failed precondition: Python interpreter state is not initialized. The process may be terminated.'
            self._sess.close()

            # Set the variable to None to avoid exceptions while closing the session again
            # in the _update_session() method.
            self._sess = None
            print('Session is closed.')

            self._create_loss_info(loss_type, save_path)
            print('Sub test is done.')

    def _create_experiment_folder(self, name):
        """
        Create folder with name `name` for saving experiments results

        """
        self._exp_folder = os.path.join(
            self._path_to_save, name
        )

        os.makedirs(self._exp_folder, exist_ok=True)

    def _update_session(self):
        # Create new session and reset the default graph.
        # It is needed to free the GPU memory from the old weights.
        print('Updating the session...')
        if self._sess is not None:
            self._sess.close()
            tf.reset_default_graph()
        self._sess = tf.Session()
        print('Session updated.')

    def _restore_model(self, exp_params):
        # Create model instance and load weights if it's needed
        print('Restoring the model...')
        # Update session before creating the model because
        # _update_session also resets the default TF graph
        # what causes deletion of every computational graph was ever built.
        self._update_session()

        # Model for train
        self._model = self._model_creation_function(
            use_gen=True,
            batch_size=exp_params[ExpField.BATCH_SIZE],
            sess=self._sess
        )

        self.generator = self._model._generator
        self.loss_list = []

        weights_path = exp_params[ExpField.WEIGHTS]
        pretrained_layers = exp_params[ExpField.PRETRAINED_LAYERS]
        untrainable_layers = exp_params[ExpField.UNTRAINABLE_LAYERS]

        self._model.set_session(self._sess)
        if weights_path is not None:
            self._model.load_weights(weights_path, layer_names=pretrained_layers)

        if untrainable_layers is not None:
            layers = []
            for layer_name in untrainable_layers:
                layers += [(layer_name, False)]
            self._model.set_layers_trainable(layers)

        # Set l1 regularization
        l1_reg = exp_params[ExpField.L1_REG]
        if l1_reg is not None:
            l1_reg = np.float32(l1_reg)
            l1_reg_layers = exp_params[ExpField.L1_REG_LAYERS]
            reg_config = [(layer, l1_reg) for layer in l1_reg_layers]
            self._model.set_l1_reg(reg_config)

        # Set l2 regularization
        l2_reg = exp_params[ExpField.L2_REG]
        if l2_reg is not None:
            l2_reg = np.float32(l2_reg)
            l2_reg_layers = exp_params[ExpField.L2_REG_LAYERS]
            reg_config = [(layer, l2_reg) for layer in l2_reg_layers]
            self._model.set_l2_reg(reg_config)

        # Model for test
        # TODO: Think about this model, may be here its not necessary to use it (Memory issue and etc...)
        self._test_model = self._model_creation_function(
            use_gen=False,
            batch_size=exp_params[ExpField.BATCH_SIZE],
            sess=self._sess
        )

        self._test_model.set_session(self._sess)
        if weights_path is not None:
            self._test_model.load_weights(weights_path, layer_names=pretrained_layers)

    def _perform_testing(self, exp_params, save_path, path_to_weights):
        # Create test video from pure model.
        # NOTICE output of model and input image size (not UV) must be equal
        # TODO: Calculate accuracy and error, plot several skelets on images (??)
        print('Testing the model...')

        # load weights to test model
        self._test_model.load_weights(path_to_weights)

        # Collect data and predictions

    def _visualize_different_graphics(self, loss_type: str, save_path: str):
        """
        Visualize loss and accuracy in training/test on certain graphics and saving them as images

        """
        TestVisualizer.plot_test_values(
            test_values=[self.loss_list],
            legends=[loss_type],
            x_label='Epochs',
            y_label='Loss',
            save_path=f'{save_path}/loss_info.png'
        )
