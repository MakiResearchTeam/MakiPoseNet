import json
import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import maximum_filter
import cv2

from .main_modules import PoseEstimatorInterface
from .utils.algorithm_connect_skelet import estimate_paf, merge_similar_skelets
from makiflow.base.maki_entities import MakiCore
from makiflow.base.maki_entities import MakiTensor
from .utils.smoother import Smoother


class PEModel(PoseEstimatorInterface):

    INPUT_MT = 'input_mt'
    OUTPUT_HEATMAP_MT = 'output_heatmap_mt'
    OUTPUT_PAF_MT = 'output_paf_mt'
    NAME = 'name'

    UPSAMPLE_SIZE = 'upsample_size'

    _DEFAULT_KERNEL_MAX_POOL = [1, 3, 3, 1]


    @staticmethod
    def from_json(path_to_model: str, input_tensor: MakiTensor = None):
        """
        Creates and returns PEModel from json file contains its architecture

        """
        # Read architecture from file
        json_file = open(path_to_model)
        json_value = json_file.read()
        json_file.close()

        json_info = json.loads(json_value)

        # Take model information
        output_heatmap_mt_names = json_info[MakiCore.MODEL_INFO][PEModel.OUTPUT_HEATMAP_MT]
        output_paf_mt_names = json_info[MakiCore.MODEL_INFO][PEModel.OUTPUT_PAF_MT]

        input_mt_name = json_info[MakiCore.MODEL_INFO][PEModel.INPUT_MT]
        model_name = json_info[MakiCore.MODEL_INFO][PEModel.NAME]
        graph_info = json_info[MakiCore.GRAPH_INFO]

        # Restore all graph variables of saved model
        inputs_and_outputs = MakiCore.restore_graph(
            output_heatmap_mt_names + output_paf_mt_names,
            graph_info,
            input_layer=input_tensor
        )

        input_x = input_tensor
        if input_x is None:
            input_x = inputs_and_outputs[input_mt_name]

        output_paf_list = [inputs_and_outputs[take_by_name] for take_by_name in output_paf_mt_names]
        output_heatmap_list = [inputs_and_outputs[take_by_name] for take_by_name in output_heatmap_mt_names]

        print('Model is restored!')

        return PEModel(
            input_x=input_x,
            output_heatmap_list=output_heatmap_list,
            output_paf_list=output_paf_list,
            name=model_name
        )

    def __init__(
        self,
        input_x: MakiTensor,
        output_paf_list: list,
        output_heatmap_list: list,
        name="Pose_estimation"
    ):
        """
        Create Pose Estimation Model which provides API to train and tests model.

        Parameters
        ----------
        input_x : MakiTensor
            Input MakiTensor
        output_paf_list : list
            List of MakiTensors which are output part affinity fields (paf).
            Assume that last tensor in the list, will be the main one
        output_heatmap_list : list
            List of MakiTensors which are output heatmaps.
            Assume that last tensor in the list, will be the main one
        name : str
            Name of this model
        """
        self.name = str(name)

        self._index_of_main_paf = len(output_paf_list) - 1

        graph_tensors = {}

        for elem in output_paf_list:
            graph_tensors.update(elem.get_previous_tensors())
            graph_tensors.update(elem.get_self_pair())

        for elem in output_heatmap_list:
            graph_tensors.update(elem.get_previous_tensors())
            graph_tensors.update(elem.get_self_pair())

        super().__init__(graph_tensors, outputs=output_paf_list + output_heatmap_list, inputs=[input_x])
        self._init_tensors_for_prediction()

    def _init_tensors_for_prediction(self):
        """
        Initialize tensors for prediction

        """
        # Store (H, W) - final size of the prediction
        self.upsample_size = tf.placeholder(dtype=tf.int32, shape=(2), name=PEModel.UPSAMPLE_SIZE)

        # [N, W, H, NUM_KP]
        main_paf = self.get_main_paf_tensor()

        # [N, W, H, NUM_PAFS, 2]
        shape_paf = main_paf.shape.as_list()
        num_pafs = shape_paf[3]

        # [N, W, H, NUM_PAFS * 2] --> [N, NEW_W, NEW_H, NUM_PAFS * 2]
        main_paf = tf.reshape(main_paf, shape=shape_paf[:-2] + [-1])
        self._resized_paf = tf.image.resize_area(
            main_paf,
            self.upsample_size,
            align_corners=False,
            name='upsample_paf'
        )

        # [N, NEW_W, NEW_H, NUM_PAFS * 2] --> [N, NEW_W, NEW_H, NUM_PAFS, 2]
        new_shape = tf.stack(
            [self.get_batch_size(), self.upsample_size[0], self.upsample_size[1], num_pafs, 2]
        )
        self._resized_paf = tf.reshape(self._resized_paf, shape=new_shape)

        self._resized_heatmap = tf.image.resize_area(
            self.get_main_heatmap_tensor(),
            self.upsample_size,
            align_corners=False,
            name='upsample_heatmap'
        )

        num_keypoints = self.get_main_heatmap_tensor().get_shape().as_list()[-1]
        self._smoother = Smoother(
            {Smoother.DATA: self._resized_heatmap},
            25,
            3.0,
            num_keypoints
        )

        # Apply NMS (Non maximum suppression)
        # Apply max pool operation to heatmap
        self._max_pooled_heatmap = tf.nn.max_pool(
            self._smoother.get_output(),
            self._DEFAULT_KERNEL_MAX_POOL,
            strides=[1,1,1,1],
            padding='SAME'
        )
        # Take only values that equal to heatmap from max pooling,
        # i.e. biggest numbers of heatmaps
        self._peaks = tf.where(
            tf.equal(
                self._smoother.get_output(),
                self._max_pooled_heatmap
            ),
            self._smoother.get_output(),
            tf.zeros_like(self._smoother.get_output())
        )

    def set_session(self, session: tf.Session):
        session.run(tf.variables_initializer(self._smoother.get_variables()))
        super().set_session(session)

    def predict(self, x: list, resize_to=None, using_estimate_alg=True):
        """
        Do pose estimation on certain input images

        Parameters
        ----------
        x : list or np.ndarray
            Input list/np.ndarray of the images
        resize_to : tuple
            Tuple of two int [H, W], which are size of the output. H - Height, W - Width.
            Resize prediction from neural network to certain size.
            By default resize not be used. If it used, by default used area interpolation
        using_estimate_alg : bool
            If equal True, when algorithm to build skeletons will be used
            And method will return list of the class Human (See Return for more detail)
            Otherwise, method will return peaks, heatmap and paf

        Returns
        -------
        if using_estimate_alg is True:
            list
                List of predictions to each input image.
                Single element of this list is a List of classes Human which were detected.

        Otherwise:
            np.ndarray
                Peaks
            np.ndarray
                Heatmap
            np.ndarray
                Paf
        """
        # Take predictions
        if resize_to is None:
            resize_to = self.get_main_paf_tensor().shape.as_list()[1:3]

        batched_heatmap, batched_paf, batched_peaks = self._session.run(
            [self._smoother.get_output(), self._resized_paf, self._peaks],
            feed_dict={
                self._input_data_tensors[0]: x,
                self.upsample_size: resize_to
            }
        )

        if using_estimate_alg:

            # Connect skeletons by applying two algorithms
            batched_humans = []

            for i in range(len(batched_peaks)):
                single_peaks = batched_peaks[i].astype(np.float32)
                single_heatmap = batched_heatmap[i].astype(np.float32)

                W, H = batched_paf[i].shape[:2]
                single_paff = batched_paf[i].reshape(W, H, -1).astype(np.float32)

                # Estimate
                humans_list = estimate_paf(single_peaks, single_heatmap, single_paff)
                # Remove similar points, simple merge similar skeletons
                humans_merged_list = merge_similar_skelets(humans_list)

                batched_humans.append(humans_merged_list)

            return batched_humans
        else:
            return batched_peaks, batched_heatmap, batched_paf

    def get_session(self):
        """
        TODO: Move this method into MakiModel
        Return tf.Session that is set for this model

        """
        return self._session

    def get_main_paf_tensor(self):

        return self._output_data_tensors[self._index_of_main_paf]

    def get_main_heatmap_tensor(self):

        return self._output_data_tensors[-1]

    def get_paf_tensors(self):
        """
        Return list of tf.Tensors which are paf (party affinity fields) calculation tensor

        """
        return self._output_data_tensors[: (self._index_of_main_paf + 1)]

    def get_paf_makitensor(self):
        """
        Return list of mf.MakiTensors which are the paf (party affinity fields) calculation tensor

        """
        return self._outputs[: (self._index_of_main_paf + 1)]

    def get_heatmap_tensors(self):
        """
        Return list of tf.Tensors which are the heatmap calculation

        """
        return self._output_data_tensors[(self._index_of_main_paf + 1):]

    def get_heatmap_makitensor(self):
        """
        Return list of mf.MakiTensor which are the heatmap calculation tensor

        """
        return self._outputs[(self._index_of_main_paf + 1):]

    def get_training_vars(self):
        """
        Return list of training variables

        """
        return self._trainable_vars

    def training_on(self):
        """
        Init variables for training

        """
        self._setup_for_training()

    def build_final_loss(self, loss):

        final_loss = self._build_final_loss(loss)

        return final_loss

    def get_batch_size(self):
        """
        Return batch size

        """
        return self._inputs[0].get_shape()[0]

    def _get_model_info(self):
        """
        Return information about model for saving architecture in JSON file

        """
        input_mt = self._inputs[0]
        output_heatmap_mt_names = [elem.get_name() for elem in self.get_heatmap_makitensor()]
        output_paf_mt_names = [elem.get_name() for elem in self.get_paf_makitensor()]

        return {
            PEModel.INPUT_MT: input_mt.get_name(),
            PEModel.OUTPUT_HEATMAP_MT: output_heatmap_mt_names,
            PEModel.OUTPUT_PAF_MT: output_paf_mt_names,
            PEModel.NAME: self.name
        }

