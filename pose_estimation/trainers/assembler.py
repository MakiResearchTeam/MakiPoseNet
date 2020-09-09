from makiflow.base import MakiTensor
from ..generators.pose_estimation import RIterator
from ..model.training_layers import BinaryHeatmapLayer


def to_makitensor(x, name):
    return MakiTensor(x, parent_layer=None, parent_tensor_names=[], previous_tensors={}, name=name)


class ModelAssembler:
    HEATMAP_CONFIG = 'HEATMAP_CONFIG'

    @staticmethod
    def __build_paf_heatmap(config, gen_layer):
        iterator = gen_layer.get_iterator()
        keypoints = iterator[RIterator.KEYPOINTS]
        masks = iterator[RIterator.KEYPOINTS_MASK]

        keypoints = to_makitensor(keypoints, 'keypoints')
        masks = to_makitensor(masks, 'masks')

        # Build heatmap layer
        heatmap_config = config[ModelAssembler.HEATMAP_CONFIG]





    @staticmethod
    def assemble(config, gen_layer, architecture):
        pass
