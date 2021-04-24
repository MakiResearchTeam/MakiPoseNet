# Copyright (C) 2020  Igor Kilbas, Danil Gribanov
#
# This file is part of MakiPoseNet.
#
# MakiPoseNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiPoseNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from ..model import PEModel, PETrainer
from ..generators.pose_estimation import RIterator
from pose_estimation.model import BinaryHeatmapLayer, GaussHeatmapLayer, V2PAFLayer, PHLabelCorrectionLayer
from makiflow.core import MakiRestorable, TrainerBuilder, MakiTensor
from makiflow.layers import InputLayer
from makiflow.distillation.core import DistillatorBuilder


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

    # Trainer config
    TRAINER_CONFIG = 'trainer_config'
    TRAINER_INFO = 'trainer_info'
    L1_REG = 'l1_reg'
    L1_REG_LAYERS = 'l1_reg_layers'
    L2_REG = 'l2_reg'
    L2_REG_LAYERS = 'l2_reg_layers'
    UNTRAINABLE_LAYERS = 'untrainable_layers'
    DISTILLATION = 'distillation_info'
    # Distillation info
    TEACHER_WEIGHTS = 'weights'
    TEACHER_ARCH = 'arch'
    # Label Correction
    LB_CONFIG = 'label_correction_config'
    LB_T_PB = 'model_pb'
    LB_INPUT_LAYER_NAME = 'input_layer_name'
    LB_PAF_LAYER_NAME = 'paf_layer_name'
    LB_HEATMAP_LAYER_NAME = 'heatmap_layer_name'
    LB_UPSAMPLE_SIZE_NAME = 'upsample_size_name'
    LB_UPSAMPLE_SIZE = 'upsample_size'

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
            config[ModelAssembler.TRAINER_CONFIG],
            model=model,
            training_paf=paf,
            training_heatmap=heatmap,
            gen_layer=gen_layer
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
        shape = gen_layer.get_shape()
        # Change batch_size to 1
        shape[0] = 1
        # Change image size to dynamic size
        shape[1] = None
        shape[2] = None
        name = gen_layer.get_name()

        input_layer = InputLayer(input_shape=shape, name=name)
        model = PEModel.from_json(
            model_config[ModelAssembler.ARCH_PATH],
            input_tensor=input_layer
        )
        model.set_session(sess)

        # Load pretrained weights
        weights_path = model_config[ModelAssembler.WEIGHTS_PATH]
        pretrained_layers = model_config[ModelAssembler.PRETRAINED_LAYERS]
        if weights_path is not None:
            model.load_weights(weights_path, layer_names=pretrained_layers)

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
        paf_layer = V2PAFLayer.build(paf_config[MakiRestorable.PARAMS])
        paf = paf_layer([keypoints, masks])

        # Setup label correction stuf
        lb_config = config.get(ModelAssembler.LB_CONFIG)
        if lb_config is not None:
            print('LABEL CORRECTION IS ON   !!!')
            input_images = iterator[RIterator.IMAGE]
            paf_heatmap_l = PHLabelCorrectionLayer(
                model_pb_path=lb_config[ModelAssembler.LB_T_PB],
                input_layer_name=lb_config[ModelAssembler.LB_INPUT_LAYER_NAME],
                paf_layer_name=lb_config[ModelAssembler.LB_PAF_LAYER_NAME],
                heatmap_layer_name=lb_config[ModelAssembler.LB_HEATMAP_LAYER_NAME],
                upsample_size_tensor_name=lb_config[ModelAssembler.LB_UPSAMPLE_SIZE_NAME],
                upsample_size=lb_config[ModelAssembler.LB_UPSAMPLE_SIZE],
            )
            # Create label correction graph with teacher NN and swap paf/heatmap with new one
            paf, heatmap = paf_heatmap_l.compile(
               input_image=input_images,
               paf_label_layer=paf,
               heatmap_label_layer=heatmap
            )
        return paf, heatmap

    @staticmethod
    def setup_trainer(config_data: dict, model: PEModel, training_paf, training_heatmap, gen_layer):
        iterator = gen_layer.get_iterator()
        absent_human_masks = iterator[RIterator.ABSENT_HUMAN_MASK]
        trainer = TrainerBuilder.trainer_from_dict(
            model=model,
            train_inputs=[gen_layer],
            label_tensors={
                PETrainer.TRAINING_HEATMAP: training_heatmap.get_data_tensor(),
                PETrainer.TRAINING_PAF: training_paf.get_data_tensor(),
                PETrainer.TRAINING_MASK: absent_human_masks
            },
            info_dict=config_data[ModelAssembler.TRAINER_INFO]
        )

        untrainable_layers = config_data[ModelAssembler.UNTRAINABLE_LAYERS]
        if untrainable_layers is not None:
            layers = []
            for layer_name in untrainable_layers:
                layers += [(layer_name, False)]
            trainer.set_layers_trainable(layers)

        # Set l1 regularization
        l1_reg = config_data[ModelAssembler.L1_REG]
        if l1_reg is not None:
            l1_reg = float(l1_reg)
            l1_reg_layers = config_data[ModelAssembler.L1_REG_LAYERS]
            reg_config = [(layer, l1_reg) for layer in l1_reg_layers]
            trainer.set_l1_reg(reg_config)

        # Set l2 regularization
        l2_reg = config_data[ModelAssembler.L2_REG]
        if l2_reg is not None:
            l2_reg = float(l2_reg)
            l2_reg_layers = config_data[ModelAssembler.L2_REG_LAYERS]
            reg_config = [(layer, l2_reg) for layer in l2_reg_layers]
            trainer.set_l2_reg(reg_config)

        distillation_config = config_data.get(ModelAssembler.DISTILLATION)
        if distillation_config is not None:
            arch_path = distillation_config[ModelAssembler.TEACHER_ARCH]
            teacher = PEModel.from_json(arch_path)
            teacher.set_session(model.get_session())

            weights_path = distillation_config[ModelAssembler.TEACHER_WEIGHTS]
            teacher.load_weights(weights_path)

            distillator = DistillatorBuilder.distillator_from_dict(
                teacher=teacher,
                info_dict=distillation_config
            )
            trainer = distillator(trainer)

        trainer.compile()
        return trainer


