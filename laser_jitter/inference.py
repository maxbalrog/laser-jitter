'''
Inference on a single time-series
'''
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (mean_absolute_error, mean_squared_error)

__all__ = ['predict_on_series']


def calculate_metrics(predictions, actuals):
    mae = nn.L1Loss(reduction='mean')(predictions, actuals)
    rms = np.sqrt(nn.MSELoss(reduction='mean')(predictions, actuals))
    # mae = mean_absolute_error(actuals, predictions)
    # rms = np.sqrt(mean_squared_error(actuals, predictions))
    return (mae, rms)


def predict_on_series(series, ts_class, model_class, device='cuda'):
    series, series_smooth = ts_class.transform_series(series)
    seq_len = len(series_smooth)
    n_features = 1 if series.ndim == 1 else series.shape[1]
    
    x = torch.Tensor(series_smooth.reshape((1,seq_len,n_features))).to(device)
    prediction = model_class.predict(x).cpu().numpy()
    prediction = ts_class.inverse_transform_series(prediction)
    return prediction

