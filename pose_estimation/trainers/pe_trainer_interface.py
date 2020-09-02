from abc import ABC, abstractmethod

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

"""
EXAMPLE OF THE TEST PARAMETERS:
experiment_params = {
    'name': name,
    'epochs': 50,
    'test period': 10,
    'save period': None or int,
    'optimizers': [
        { 
            type: ...
            params: {}
        }, 
        { 
            type: ...
            params: {}
        }
    ]
    'loss type': 'HardLoss' or 'other_name_loss',
    'batch sizes': [10],
    'weights': ../weights/weights.ckpt,
    'path to arch': path,
    'metric type': 'SPEC_OKS',
    'pretrained layers': [layer_name],
    'utrainable layers': [layer_name],
    'l1 reg': 1e-6 or None,
    'l1 reg layers': [layer_name],
    'l2 reg': 1e-6 or None,
    'l2 reg layers': [layer_name]
}"""


class ExpField:
    EXPERIMENTS = 'experiments'
    NAME = 'name'
    PRETRAINED_LAYERS = 'pretrained layers'
    WEIGHTS = 'weights'
    UNTRAINABLE_LAYERS = 'untrainable layers'
    EPOCHS = 'epochs'
    TEST_PERIOD = 'test period'
    LOSS_TYPE = 'loss type'
    METRIC_TYPE = 'metric type'
    L1_REG = 'l1 reg'
    L1_REG_LAYERS = 'l1 reg layers'
    L2_REG = 'l2 reg'
    L2_REG_LAYERS = 'l2 reg layers'
    OPTIMIZERS = 'optimizers'
    ITERATIONS = 'iterations'
    PATH_TEST_IMAGE = 'path_test_image'
    BATCH_SIZE = 'batch_size'


class SubExpField:
    OPT_INFO = 'opt_info'


class LossType:
    HARD_LOSS = 'HardLoss'


class PETrainerInterface(ABC):
    def __init__(self, model_creation_function, exp_params: str, path_to_save: str):
        # Load parameters for experiments from json file
        pass

    @abstractmethod
    def start_experiments(self):
        # Start experiments and init all necessary parameters for each experiment
        pass

    @abstractmethod
    def _run_experiment(self, exp_params):
        # Main loop of the model training for single experiment
        pass

    @abstractmethod
    def _restore_model(self, exp_params: dict):
        # Create model instance and load weights if it's needed
        # Init/update session for experiments and etc connected with model
        pass

    @abstractmethod
    def _perform_testing(self, exp_params, save_path, path_to_weights):
        # Calculate test error and accuracy of the model
        # TODO: Test different metrics for measure accuracy of the model
        pass

    @abstractmethod
    def _visualize_different_graphics(self, loss_type: str, save_path: str):
        # Visualize loss and accuracy in training/test on certain graphics and saving them as images
        pass

