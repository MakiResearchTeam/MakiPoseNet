from ..model import PEModel, MSETrainer
from ..generators.pose_estimation import RIterator
from ..model.training_layers import BinaryHeatmapLayer, GaussHeatmapLayer, PAFLayer
from makiflow.base import MakiRestorable
from makiflow.base import MakiTensor


def to_makitensor(x, name):
    return MakiTensor(x, parent_layer=None, parent_tensor_names=[], previous_tensors={}, name=name)


class ModelAssembler:
    # heatmap config: must contain `params` field
    HEATMAP_CONFIG = 'heatmap_config'
    GAUSSIAN = 'gaussian'

    # paf config: must contain `params` field
    PAF_CONFIG = 'paf_config'

    # model config
    MODEL_CONFIG = 'model_config'
    ARCH_PATH = 'arch_path'
    WEIGHTS_PATH = 'weights_path'
    PRETRAINED_LAYERS = 'pretrained_layers'
    UNTRAINABLE_LAYERS = 'untrainable_layers'
    L1_REG = 'l1_reg'
    L1_REG_LAYERS = 'l1_reg_layers'
    L2_REG = 'l2_reg'
    L2_REG_LAYERS = 'l2_reg_layers'

    # Trainer config
    LOSS = 'loss'
    PARAMETERS = 'parameters'

    # gen_layer config
    GENLAYER_CONFIG = 'genlayer_config'
    TFRECORDS_PATH = 'tfrecords_path'
    IM_HW = 'im_hw'
    PREFETCH_SIZE = 'prefetch_size'
    BATCH_SZ = 'batch_size'
    KP_SHAPE = 'keypoints_shape'

    @staticmethod
    def assemble(config, gen_layer_fabric, sess):
        gen_layer = ModelAssembler.build_gen_layer(config[ModelAssembler.GENLAYER_CONFIG], gen_layer_fabric)
        model = ModelAssembler.setup_model(config[ModelAssembler.MODEL_CONFIG], gen_layer, sess)
        paf, heatmap = ModelAssembler.build_paf_heatmap(config, gen_layer)
        trainer = ModelAssembler.setup_trainer(
            config.get(ModelAssembler.LOSS),
            model=model,
            training_paf=paf,
            training_heatmap=heatmap
        )
        return trainer, model

    @staticmethod
    def build_gen_layer(config, gen_layer_fabric):
        return gen_layer_fabric(
            tfrecords_path=config[ModelAssembler.TFRECORDS_PATH],
            im_hw=config[ModelAssembler.IM_HW],
            batch_sz=config[ModelAssembler.BATCH_SZ],
            prefetch_sz=config[ModelAssembler.PREFETCH_SIZE],
            kp_shape=config[ModelAssembler.KP_SHAPE]
        )

    @staticmethod
    def setup_model(model_config, gen_layer, sess):
        model = PEModel.from_json(model_config[ModelAssembler.ARCH_PATH], gen_layer)
        model.set_session(sess)

        # Load pretrained weights
        weights_path = model_config[ModelAssembler.WEIGHTS_PATH]
        pretrained_layers = model_config[ModelAssembler.PRETRAINED_LAYERS]
        if weights_path is not None:
            model.load_weights(weights_path, layer_names=pretrained_layers)

        untrainable_layers = model_config[ModelAssembler.UNTRAINABLE_LAYERS]
        if untrainable_layers is not None:
            layers = []
            for layer_name in untrainable_layers:
                layers += [(layer_name, False)]
            model.set_layers_trainable(layers)

        # Set l1 regularization
        l1_reg = model_config[ModelAssembler.L1_REG]
        if l1_reg is not None:
            l1_reg = float(l1_reg)
            l1_reg_layers = model_config[ModelAssembler.L1_REG_LAYERS]
            reg_config = [(layer, l1_reg) for layer in l1_reg_layers]
            model.set_l1_reg(reg_config)

        # Set l2 regularization
        l2_reg = model_config[ModelAssembler.L2_REG]
        if l2_reg is not None:
            l2_reg = float(l2_reg)
            l2_reg_layers = model_config[ModelAssembler.L2_REG_LAYERS]
            reg_config = [(layer, l2_reg) for layer in l2_reg_layers]
            model.set_l2_reg(reg_config)

        return model

    @staticmethod
    def build_paf_heatmap(config, gen_layer):
        # Extract tensors with keypoints and their masks
        iterator = gen_layer.get_iterator()
        keypoints = iterator[RIterator.KEYPOINTS]
        masks = iterator[RIterator.KEYPOINTS_MASK]

        keypoints = to_makitensor(keypoints, 'keypoints')
        masks = to_makitensor(masks, 'masks')

        # Build heatmap layer
        heatmap_config = config[ModelAssembler.HEATMAP_CONFIG]
        if heatmap_config[ModelAssembler.GAUSSIAN]:
            layer = GaussHeatmapLayer
        else:
            layer = BinaryHeatmapLayer
        heatmap_layer = layer.build(heatmap_config[MakiRestorable.PARAMS])
        heatmap = heatmap_layer([keypoints, masks])

        # Build paf layer
        paf_config = config[ModelAssembler.PAF_CONFIG]
        paf_layer = PAFLayer.build(paf_config[MakiRestorable.PARAMS])
        paf = paf_layer([keypoints, masks])

        return paf, heatmap

    @staticmethod
    def setup_trainer(config_data: dict, model: PEModel, training_paf, training_heatmap):
        # TODO: Add separate class to build trainers
        trainer = MSETrainer(
            model=model,
            training_paf=training_paf,
            training_heatmap=training_heatmap
        )

        if config_data is not None:
            config_trainer = config_data[ModelAssembler.PARAMETERS]

            heatmap_scale = config_trainer[MSETrainer.HEATMAP_SCALE]
            paf_scale = config_trainer[MSETrainer.PAF_SCALE]

            heatmap_single_scale = config_trainer[MSETrainer.HEATMAP_SINGLE_SCALE]
            paf_single_scale = config_trainer[MSETrainer.PAF_SINGLE_SCALE]

            trainer.set_loss_scales(
                paf_scale=paf_scale,
                heatmap_scale=heatmap_scale,
                heatmap_single_scale=heatmap_single_scale,
                paf_single_scale=paf_single_scale
            )

        return trainer


