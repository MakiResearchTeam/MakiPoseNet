import tensorflow as tf
from pose_estimation.metrics.COCO_WholeBody import relayout_keypoints
from abc import ABC, abstractmethod
import os
from pycocotools.coco import COCO


class Tester(ABC):
    # Using in conjugation with trainer.
    # After the model was trained for some time, call the evaluation
    # method and all the info will recorded to the tensorboard.
    TEST_CONFIG = 'test_config'
    TB_FOLDER = 'tb_folder'  # folder for tensorboard to write data in
    TEST_IMAGE = 'test_image'
    BATCH_SIZE = 'batch_size'
    LOG_FOLDER = 'logs'
    ANNOT_GT_JSON = 'annot_gt_json'
    PATH_TO_VAL_IMAGES = "path_to_val_images"
    LIMIT_ANNOT = 'limit_annot'
    N_THREADE = 'n_threade'
    TYPE_PARALL = 'type_parall'
    NORMALIZATION_SHIFT = 'normalization_shift'
    NORMALIZATION_DIV = 'normalization_div'
    NORM_MODE = 'norm_mode'
    USE_BGR2RGB = 'use_bgr2rgb'
    IMG_HW = 'img_hw'

    NAME_RELAYOUR_ANNOT_JSON = "relayour_annot.json"
    NAME_PREDICTED_ANNOT_JSON = 'predicted_annot.json'
    AP_AR_DATA_TXT = 'ap_ar_data.txt'

    def __init__(self, config: dict, sess, path_to_save_logs:str):
        self._config = config[Tester.TEST_CONFIG]

        self._path_to_save_logs = os.path.join(path_to_save_logs, self.LOG_FOLDER)
        os.makedirs(self._path_to_save_logs, exist_ok=True)

        self._path_to_relayout_annot = os.path.join(self._path_to_save_logs, self.NAME_RELAYOUR_ANNOT_JSON)

        self._tb_writer = tf.summary.FileWriter(config[Tester.TB_FOLDER])
        self._sess = sess

        # Init stuff for measure metric
        self._limit_annots = self._config[self.LIMIT_ANNOT]
        self._n_threade = self._config[self.N_THREADE]
        self._type_parall = self._config[self.TYPE_PARALL]

        self._norm_div = self._config[self.NORMALIZATION_DIV]
        self._norm_shift = self._config[self.NORMALIZATION_SHIFT]

        self._norm_mode = self._config[self.NORM_MODE]
        self._use_bgr2rgb = self._config[self.USE_BGR2RGB]
        self.W = self._config[Tester.IMG_HW][1]
        self.H = self._config[Tester.IMG_HW][0]

        annot_gt = self._config[self.ANNOT_GT_JSON]

        if annot_gt is not None:
            relayout_keypoints(
                self.W, self.H,
                self._config[self.ANNOT_GT_JSON], self._path_to_relayout_annot,
                self._limit_annots
            )

            # Load ground-truth annot
            self.cocoGt = COCO(self._path_to_relayout_annot)
            self._path_to_val_images = self._config[self.PATH_TO_VAL_IMAGES]
        else:
            self.cocoGt = None

        # The summaries to write
        self._summaries = {}
        # Placeholder that take in the data for the summary
        self._summary_inputs = {}

        self._init(self._config)

    def _init(self, config):
        pass

    def add_image(self, name, n_images=1):
        """
        Adds an image summary to the tensorboard.
        The image dtype must by uint8 and have shape (batch_size, h, w, c).

        Parameters
        ----------
        name : str
            Name that will be displayed on the tensorboard.
        n_images : int
            Maximum number of images to display on the board.
        """
        image = tf.placeholder(dtype=tf.uint8)
        self._summary_inputs.update(
            {name: image}
        )
        image_summary = tf.summary.image(name, image, max_outputs=n_images)
        self._summaries.update(
            {name: image_summary}
        )

    def add_scalar(self, name):
        """
        Adds a scalar summary (e.g. accuracy) to the tensorboard.
        The image dtype must by float32.

        Parameters
        ----------
        name : str
            Name that will be displayed on the tensorboard.
        """
        scalar = tf.placeholder(dtype=tf.float32)
        self._summary_inputs.update(
            {name: scalar}
        )
        scalar_summary = tf.summary.scalar(name, scalar)
        self._summaries.update(
            {name: scalar_summary}
        )

    def write_summaries(self, summaries, step=None):
        """
        Writes the summary to the tensorboard log file.
        Parameters
        ----------
        summaries : dict
            Contains pairs (name, data). `data` can be whether scalar or image.
        step : int
            The training/evaluation step number.
        """
        for summary_name in summaries:
            data = summaries[summary_name]
            s_input = self._summary_inputs[summary_name]
            summary = self._summaries[summary_name]

            summary_tensor = self._sess.run(
                summary,
                feed_dict={
                    s_input: data
                }
            )
            self._tb_writer.add_summary(summary_tensor, global_step=step)
        # self._tb_writer.flush()

    @abstractmethod
    def evaluate(self, model, iteration):
        pass

    def get_writer(self):
        return self._tb_writer
