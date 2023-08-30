'''
This script provides higher-level interface for NN models from model_basic.py including training models,
inference on dataloaders and single time_series
'''

import numpy as np
import torch
import torch.nn as nn

__all__ = ['RNNTemporal']


class RNNTemporal:
    def __init__(self, model, model_params, save_path=''):
        '''

        '''
        self.model = model
        self.model_params = model_params
        self.save_path = save_path

    def train(self):
        pass

    def inference_on_dataloader():
        pass

    def predict_on_series():
        pass

