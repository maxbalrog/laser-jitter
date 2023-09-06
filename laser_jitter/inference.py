'''
Inference on a single time-series
'''
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (mean_absolute_error, mean_squared_error)

__all__ = ['calculate_metrics']


def calculate_metrics(predictions, actuals):
    mae = nn.L1Loss(reduction='mean')(predictions, actuals)
    rms = np.sqrt(nn.MSELoss(reduction='mean')(predictions, actuals))
    return (mae, rms)

