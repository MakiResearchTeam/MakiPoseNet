from .assembler import ModelAssembler
import json
import os
from makiflow.trainers.utils.optimizer_builder import OptimizerBuilder
from .tester import Tester
from .coco_tester import CocoTester


class PEGym:
    MSG_CREATE_FOLDER = 'Creating gym folder...'
    WRN_GYM_FOLDER_EXISTS = 'Warning! Gym folder with the specified path={0} already' \
                            'exists. Creating another directory with path={1}.'
    """
    Config file consists of several main sections:
    - heatmap_config
    - paf_config
    - model_config
    - training_config
    - testing_config
    """
    TRAIN_CONFIG = 'train_config'
    EPOCHS = 'epochs'
    ITERS = 'iters'
    TEST_PERIOD = 'test_period'
    SAVE_PERIOD = 'save_period'
    PRINT_PERIOD = 'print_period'
    GYM_FOLDER = 'gym_folder'
    OPTIMIZER_INFO = 'optimizer_info'

    def __init__(self, config_path, gen_layer, sess):
        with open(config_path) as json_file:
            json_value = json_file.read()
            config = json.loads(json_value)

        self._train_config = config[PEGym.TRAIN_CONFIG]
        self._gen_layer = gen_layer
        self._sess = sess
        self._setup_gym(config)

    def _setup_gym(self, config):
        self._create_gym_folder()

        # Create folder for the last weights of the model
        self._last_w_folder_path = os.path.join(
            self._train_config[PEGym.GYM_FOLDER],
            'last_weights'
        )
        os.makedirs(self._last_w_folder_path, exist_ok=True)

        # Create folder for the tensorboard and create tester
        tensorboard_path = os.path.join(
            self._train_config[PEGym.GYM_FOLDER],
            'tensorboard'
        )
        os.makedirs(tensorboard_path, exist_ok=True)
        config[Tester.TB_FOLDER] = tensorboard_path
        self._tb_path = tensorboard_path
        self._tester = CocoTester(config, self._sess)

        # Create model, trainer and set the tensorboard folder
        self._trainer, self._model = ModelAssembler.assemble(config, self._gen_layer, self._sess)
        self._trainer.set_tensorboard_writer(self._tester.get_writer())

    def get_tb_path(self):
        return self._tb_path

    def _create_gym_folder(self):
        print(PEGym.MSG_CREATE_FOLDER)
        orig_path = self._train_config[PEGym.GYM_FOLDER]
        path = orig_path
        counter = 1
        while os.path.isdir(path):
            new_path = orig_path + f'_{counter}'
            counter += 1
            print(PEGym.WRN_GYM_FOLDER_EXISTS.format(path, new_path))
            path = new_path

        os.makedirs(path, exist_ok=True)

    def get_model(self):
        """
        May be used for adding a custom loss to the model.
        """
        return self._model

    def start_training(self):
        epochs = self._train_config[PEGym.EPOCHS]
        iters = self._train_config[PEGym.ITERS]
        test_period = self._train_config[PEGym.TEST_PERIOD]
        save_period = self._train_config[PEGym.SAVE_PERIOD]
        print_period = self._train_config[PEGym.PRINT_PERIOD]

        optimizer, global_step = OptimizerBuilder.build_optimizer(
            self._train_config[PEGym.OPTIMIZER_INFO]
        )

        it_counter = 0
        for i in range(1, epochs + 1):
            _ = self._trainer.fit(
                optimizer=optimizer, epochs=1, iter=iters, global_step=global_step, print_period=print_period
            )
            it_counter += iters

            if i % test_period == 0:
                self._tester.evaluate(self._model, it_counter)

            if i % save_period == 0:
                self._save_weights(i)

        path_to_save = os.path.join(
            self._last_w_folder_path, 'weights.ckpt'
        )
        self._model.save_weights(path_to_save)

    def _save_weights(self, epoch):
        gym_folder = self._train_config[PEGym.GYM_FOLDER]
        save_path = os.path.join(
            gym_folder, f'epoch_{epoch}'
        )
        os.makedirs(save_path, exist_ok=True)
        self._model.save_weights(save_path)




