import json

from makiflow.base.maki_entities import MakiCore
from makiflow.base.maki_entities import MakiTensor


class PEModel(MakiCore):

    INPUT_MT = 'input_mt'
    OUTPUT_HEATMAP_MT = 'output_heatmap_mt'
    OUTPUT_PAFF_MT = 'output_paff_mt'
    NAME = 'name'

    @staticmethod
    def from_json(path_to_model: str):
        """
        Creates and returns PEModel from json file contains its architecture

        """
        json_file = open(path_to_model)
        json_value = json_file.read()
        json_info = json.loads(json_value)

        output_heatmap_mt_name = json_info[MakiCore.MODEL_INFO][PEModel.OUTPUT_HEATMAP_MT]
        output_paff_mt_name = json_info[MakiCore.MODEL_INFO][PEModel.OUTPUT_PAFF_MT]

        input_mt_name = json_info[MakiCore.MODEL_INFO][PEModel.INPUT_MT]
        model_name = json_info[MakiCore.MODEL_INFO][PEModel.NAME]

        graph_info = json_info[MakiCore.GRAPH_INFO]

        inputs_and_outputs = MakiCore.restore_graph(
            [output_heatmap_mt_name, output_paff_mt_name],
            graph_info
        )

        input_x = inputs_and_outputs[input_mt_name]

        output_paf = inputs_and_outputs[output_paff_mt_name]
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

    def get_paff_tensor(self):
        """
        Return tf.Tensor of paf calculation

        """
        return self._output_data_tensors[0]

    def get_heatmap_tensor(self):
        """
        Return tf.Tensor of heatmap calculation

        """
        return self._output_data_tensors[1]

    def get_training_vars(self):
        """
        Return list of training variables

        """
        return self._trainable_vars

    def build_final_loss(self, loss):

        final_loss = self.build_final_loss(loss)

        return final_loss

    def _get_model_info(self):
        """
        Return information about model for saving architecture in JSON file

        """
        input_mt = self._inputs[0]
        output_heatmap_mt = self.get_heatmap_tensor()
        output_paff_mt = self.get_paff_tensor()

        return {
            PEModel.INPUT_MT: input_mt.get_name(),
            PEModel.OUTPUT_HEATMAP_MT: output_heatmap_mt.get_name(),
            PEModel.OUTPUT_PAFF_MT: output_paff_mt.get_name(),
            PEModel.NAME: self.name
        }