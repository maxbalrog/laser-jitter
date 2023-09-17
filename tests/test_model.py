import numpy as np
import torch
import torch.nn as nn
import os

from laser_jitter.data import (TimeSeries, TimeSeriesSTFT)
from laser_jitter.model_basic import LSTMForecaster
from laser_jitter.model import (RNN_abc, RNNTemporal, RNNSTFT,
                                RNNSTFTInTimeOut, RNNSTFT_real_imag,
                                RNNSTFT_ensemble)


n_features = 1
n_hidden_lstm = 64
training_window = 300
prediction_window = 100
use_cuda = False #torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

model_params = {
    'n_features': n_features,
    'n_hidden_lstm': n_hidden_lstm,
    'n_hidden_fc': 100,
    'n_outputs': prediction_window*n_features,
    'n_out_features': n_features,
    'sequence_len': training_window,
    'n_lstm_layers': 2,
    'n_deep_layers': 1,
    'dropout': 0.2,
    'use_cuda': use_cuda
}
save_folder = 'model_/'

batch_size = 16
x = torch.rand((batch_size,training_window,n_features))

N_sm = 5
smooth_params = {
    'kernel': np.ones(N_sm)/N_sm,
}
scaling = 'standard'
train_size = 0.8

N = 2000
series = np.random.uniform(size=(N))
series_class = TimeSeries(series, smooth_params, scaling, train_size)

window_length = 400
stft_params = {
    'nperseg': window_length,
    'fs': 1e3,
    'boundary': None,
    'noverlap': window_length-1,
    'padded': False,
    'window': 'tukey'
}

filter_params = {
    'thresh_weight': 1,
    'freq_low': 0
}
series_stft_class = TimeSeriesSTFT(series, stft_params, scaling, train_size,
                                   filter_params, smooth_params=None)

sequence_params = {
    'training_window': training_window,
    'prediction_window': prediction_window,
    'step': 1
}

dataloader_params = {
    'batch_size': 128,
    'drop_last': False,
}

loaders = series_class.create_dataloaders(series_class.train_smooth,
                                          series_class.test_smooth,
                                          sequence_params, dataloader_params)
trainloader_smooth, testloader_smooth = loaders


def test_RNN_abs():
    model = LSTMForecaster(**model_params)
    for model_ in [model, None]:
        for load_model in [False, True]:
            model_abc = RNN_abc(model_params, model_, save_folder, load_model=load_model)
    model_abc.save_model_params()
    params = model_abc.load_model_params()
    assert params == model_params
    os.remove(f'{os.path.dirname(save_folder)}/model_params.yml')
    os.rmdir(save_folder)


def test_RNNTemporal():
    model = LSTMForecaster(**model_params)
    for model_ in [model, None]:
        for load_model in [False, True]:
            model_abc = RNNTemporal(model_params, model_, save_folder, load_model=load_model)
    model_abc.save_model_params()
    params = model_abc.load_model_params()
    assert params == model_params

    series_slice = series[:training_window+N_sm//2*2]
    out = model_abc.predict_on_series(series_slice, series_class, device)
    
    os.remove(f'{os.path.dirname(save_folder)}/model_params.yml')
    os.rmdir(save_folder)


def test_RNNSTFT():
    n_features = len(series_stft_class.freq_filt)*2
    model_params_ = model_params.copy()
    model_params_['n_features'] = n_features
    model_params_['n_out_features'] = n_features
    model_params_['n_outputs'] = prediction_window*n_features
    
    model = LSTMForecaster(**model_params)
    for model_ in [model, None]:
        for load_model in [False, True]:
            model_abc = RNNSTFT(model_params_, model_, save_folder, load_model=load_model)
    model_abc.save_model_params()
    params = model_abc.load_model_params()
    assert params == model_params_

    series_slice = series[:training_window+window_length-1]
    out = model_abc.predict_on_series(series_slice, series_stft_class, device)
    
    os.remove(f'{os.path.dirname(save_folder)}/model_params.yml')
    os.rmdir(save_folder)


def test_RNNSTFTInTimeOut():
    pass


def test_RNNSTFT_real_imag():
    n_features = len(series_stft_class.freq_filt)
    model_params_ = model_params.copy()
    model_params_['n_features'] = n_features
    model_params_['n_out_features'] = n_features
    model_params_['n_outputs'] = prediction_window*n_features
    
    model = None
    for model_ in [model, None]:
        for load_model in [False, True]:
            model_abc = RNNSTFT_real_imag(model_params_, model_, save_folder,
                                          load_model=load_model)
    model_abc.save_model_params()
    params = model_abc.load_model_params()
    assert params == model_params_

    series_slice = series[:training_window+window_length-1]
    out = model_abc.predict_on_series(series_slice, series_stft_class, device)
    
    os.remove(f'{os.path.dirname(save_folder)}/model_params.yml')
    os.rmdir(save_folder)


def test_RNNSTFT_ensemble():
    n_models = len(series_stft_class.freq_filt)
    n_features = 2
    model_params_ = model_params.copy()
    model_params_['n_features'] = n_features
    model_params_['n_out_features'] = n_features
    model_params_['n_outputs'] = prediction_window*n_features
    
    model = None
    for model_ in [model, None]:
        for load_model in [False, True]:
            model_abc = RNNSTFT_ensemble(model_params_, model_, save_folder,
                                          load_model=load_model, n_models=n_models)
    model_abc.save_model_params()
    params = model_abc.load_model_params()
    assert params == model_params_

    series_slice = series[:training_window+window_length-1]
    out = model_abc.predict_on_series(series_slice, series_stft_class, device)
    
    os.remove(f'{os.path.dirname(save_folder)}/model_params.yml')
    os.rmdir(save_folder)





