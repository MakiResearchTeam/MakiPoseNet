import json
import numpy as np
from scipy.ndimage.filters import maximum_filter

from .main_modules import PoseEstimatorInterface
from .utils.algorithm_connect_skelet import estimate_paf, merge_similar_skelets
from makiflow.base.maki_entities import MakiCore
from makiflow.base.maki_entities import MakiTensor
from makiflow.base.maki_entities import InputMakiLayer


class PEModel(PoseEstimatorInterface):

    INPUT_MT = 'input_mt'
    OUTPUT_HEATMAP_MT = 'output_heatmap_mt'
    OUTPUT_PAF_MT = 'output_paf_mt'
    NAME = 'name'

    @staticmethod
    def from_json(path_to_model: str, input_tensor: MakiTensor):
        """
        Creates and returns PEModel from json file contains its architecture

        """
        # Read architecture from file
        json_file = open(path_to_model)
        json_value = json_file.read()
        json_file.close()

        json_info = json.loads(json_value)

        # Take model information
        output_heatmap_mt_name = json_info[MakiCore.MODEL_INFO][PEModel.OUTPUT_HEATMAP_MT]
        output_paf_mt_name = json_info[MakiCore.MODEL_INFO][PEModel.OUTPUT_PAF_MT]
        input_mt_name = json_info[MakiCore.MODEL_INFO][PEModel.INPUT_MT]
        model_name = json_info[MakiCore.MODEL_INFO][PEModel.NAME]
        graph_info = json_info[MakiCore.GRAPH_INFO]

        # Restore all graph variables of saved model
        inputs_and_outputs = MakiCore.restore_graph(
            [output_heatmap_mt_name, output_paf_mt_name],
            graph_info,
            input_layer=input_tensor
        )

        input_x = inputs_and_outputs[input_mt_name]

        output_paf = inputs_and_outputs[output_paf_mt_name]
        output_heatmap = inputs_and_outputs[output_heatmap_mt_name]

        print('Model is restored!')

        return PEModel(
            input_x=input_x,
            output_heatmap=output_heatmap,
            output_paf=output_paf,
            name=model_name
        )

    def __init__(
        self,
        input_x: MakiTensor,
        output_paf: MakiTensor,
        output_heatmap: MakiTensor,
        name="Pose_estimation"
    ):
        """
        Create Pose Estimation Model which provides API to train and tests model.

        Parameters
        ----------
        input_x : MakiTensor
            Input MakiTensor
        output_paf : MakiTensor
            Output part affinity fields (paf) MakiTensor
        output_heatmap : MakiTensor
            Output heatmap MakiTensor
        name : str
            Name of this model
        """
        self.name = str(name)

        graph_tensors = output_paf.get_previous_tensors()
        graph_tensors.update(output_paf.get_self_pair())

        graph_tensors.update(output_heatmap.get_previous_tensors())
        graph_tensors.update(output_heatmap.get_self_pair())
        super().__init__(graph_tensors, outputs=[output_paf, output_heatmap], inputs=[input_x])

    def predict(self, x: list, pooling_window_size=(3, 3), using_estimate_alg=True):
        """
        Do pose estimation on certain input images

        Parameters
        ----------
        x : list or np.ndarray
            Input list/np.ndarray of the images
        pooling_window_size : tuple
            Size of the pooling window,
            By default equal to (3, 3)
        using_estimate_alg : bool
            If equal True, when algorithm to build skeletons will be used
            And method will return list of the class Human (See Return for more detail)
            Otherwise, method will return peaks, heatmap and paf

        Returns
        -------
        if using_estimate_alg is True:
            list
                List of classes Human.
                Human class store set of `body_parts` to each keypoints detected by neural network,
                Set `body_parts` is set of classes BodyPart, set itself is store values from 0 to n
                (n - number of keypoints on the skeleton)
                which were detected by neural network (i.e. some of keypoints may not be present).
                To get x and y coordinate, take one of the `body_parts` element (which is BodyPart class),
                and get x or y by referring to the corresponding field.
        Otherwise:
            np.ndarray
                Peaks
            np.ndarray
                Heatmap
            np.ndarray
                Paf
        """
        # Take predictions
        batched_heatmap, batched_paf = self._session.run(
            [self.get_heatmap_tensors(), self.get_paf_tensors()],
            feed_dict={self._input_data_tensors[0]: x}
        )

        # Apply NMS (Non maximum suppression)
        batched_max_pool_heatmap = maximum_filter(
            batched_heatmap,
            # Footprint goes on shape (N, W, H, C), skip window for N and C dimensions
            footprint=np.ones((1, pooling_window_size[0], pooling_window_size[1], 1))
        )
        # Take only "peaks" i. e. most probably keypoints on the image
        batched_peaks = batched_heatmap * (batched_max_pool_heatmap == batched_heatmap)

        # Connect skeletons by applying two algorithms
        batched_humans = []

        if using_estimate_alg:
            W, H = self._inputs[0].get_shape()[1:3]

            for i in range(len(batched_peaks)):
                single_peaks = batched_peaks[i].astype(np.float32)
                single_heatmap = batched_heatmap[i].astype(np.float32)
                single_paff = batched_paf[i].reshape(W, H, -1).astype(np.float32)
                # Estimate
                humans = estimate_paf(single_peaks, single_heatmap, single_paff)
                # Remove similar points, simple merge similar skeletons
                humans_dict = merge_similar_skelets(humans)

                batched_humans.append(humans_dict)

            return batched_humans
        else:
            return batched_peaks, batched_heatmap, batched_paf

    def get_session(self):
        """
        TODO: Move this method into MakiModel
        Return tf.Session that is set for this model

        """
        return self._session

    def get_paf_tensors(self):
        """
        Return tf.Tensor of paf (party affinity fields) calculation

        """
        return self._output_data_tensors[0]

    def get_paf_makitensor(self):
        """
        Return mf.MakiTensor of the paf (party affinity fields) calculation tensor

        """
        return self._outputs[0]

    def get_heatmap_tensors(self):
        """
        Return tf.Tensor of heatmap calculation

        """
        return self._output_data_tensors[1]

    def get_heatmap_makitensor(self):
        """
        Return mf.MakiTensor of the heatmap calculation tensor

        """
        return self._outputs[1]

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
        output_heatmap_mt = self.get_heatmap_makitensor()
        output_paf_mt = self.get_paf_makitensor()

        return {
            PEModel.INPUT_MT: input_mt.get_name(),
            PEModel.OUTPUT_HEATMAP_MT: self._outputs[1].get_name(),
            PEModel.OUTPUT_PAF_MT: self._outputs[0].get_name(),
            PEModel.NAME: self.name
        }

