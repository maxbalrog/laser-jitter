'''
Higher-level interface for NN models from model_basic.py including prediction with models,
inference on dataloaders and single time_series
'''
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from laser_jitter.data import create_dataloader
from laser_jitter.model_basic import LSTMForecaster
from laser_jitter.utils import read_yaml, write_yaml
from laser_jitter.train import train_model, train_model_real_imag
from laser_jitter.inference import predict_on_series, calculate_metrics

__all__ = ['RNNTemporal', 'RNNSTFT']


class RNNTemporal:
    def __init__(self, model_params=None, model=None, save_path='', SEED=23):
        '''
        High-level abstraction class for using NN model on temporal data
        to do time-series prediction

        model_params: [dict] - parameters of NN model
        model: [laser_jitter.model_basic.LSTMForecaster] - model used for prediction
        save_path: [str] - path for model weights and model info to be saved
        '''
        self.model_params = model_params
        self.device = 'cuda' if model_params['use_cuda'] else 'cpu'
        self.save_path = save_path
        
        self.model = model if model is not None else self.load_best_model()
        self.model.to(self.device)
        
        # make sure that directory for saving the model exists
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        self.save_model_params()

        self.SEED = SEED

    def save_model_params(self):
        self.yaml_file = f'{os.path.dirname(self.save_path)}/model_params.yml'
        write_yaml(self.yaml_file, self.model_params)

    def load_model_params(self):
        model_params = read_yaml(self.yaml_file)
        return model_params

    def load_best_model(self):
        self.model = LSTMForecaster(**self.model_params).to(self.device)
        self.model.load_state_dict(torch.load(self.save_path))
        return self.model

    def predict(self, x):
        with torch.no_grad():
            prediction = self.model(x).squeeze()
        return prediction

    def train(self, trainloader, testloader, criterion, optimizer, n_epochs=30, verbose=True):
        self.model = self.model.to(self.device)
        losses = train_model(self.model, trainloader, testloader, criterion, optimizer,
                             n_epochs=n_epochs, SEED=self.SEED, save_path=self.save_path,
                             device=self.device, verbose=True, model_params=self.model_params)
        return losses

    def inference_on_dataloader(self, model, dataloader, dataloader_smooth=None, device='cuda'):
        model = model.to(device)
        model.eval()
        predictions, actuals, actuals_smooth = [], [], []
        if dataloader_smooth is None:
            for x, y in dataloader:
                with torch.no_grad():
                    p = model(x.to(device)).cpu()
                    predictions.append(p)
                    actuals.append(y)
        else:
            for (x,y), (x_smooth, y_smooth) in zip(dataloader, dataloader_smooth):
                with torch.no_grad():
                    p = model(x_smooth.to(device)).cpu()
                    predictions.append(p)
                    actuals.append(y)
                    actuals_smooth.append(y_smooth)
        predictions = torch.cat(predictions).squeeze()
        actuals = torch.cat(actuals).squeeze()
        actuals_smooth = torch.cat(actuals_smooth).squeeze()

        metrics = calculate_metrics(predictions.flatten(), actuals.flatten())
        
        return (predictions.numpy(), actuals.numpy(), actuals_smooth.numpy()), metrics

    def predict_on_series(self, series, series_class, device='cuda'):
        series, series_smooth = series_class.transform_series(series)
        seq_len = len(series_smooth)
        n_features = 1 if series.ndim == 1 else series.shape[1]
        
        x = torch.Tensor(series_smooth.reshape((1,seq_len,n_features))).to(device)
        prediction = self.predict(x).cpu().numpy()
        prediction = series_class.inverse_transform_series(prediction)
        return prediction

class RNNSTFT:
    def __init__(self, model_params=None, model=None, save_path='', load_model=False, SEED=23):
        '''
        High-level abstraction class for using NN model on spectral data
        to do time-series prediction

        model_params: [dict] - parameters of NN model
        model: [laser_jitter.model_basic.LSTMForecaster] - model used for prediction
        save_path: [str] - path for model weights and model info to be saved
        '''
        self.model_params = model_params
            
        self.device = 'cuda' if model_params['use_cuda'] else 'cpu'
        self.save_path = save_path
        self.save_paths = self.create_save_paths()
       
        # make sure that directory for saving the model exists
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        self.save_model_params()
        
        if model is None:
            self.create_model()
            if load_model and self.check_save_path():
                self.load_best_model()
        else:
            self.model = model

        self.SEED = SEED
        self.n_features = model_params['n_features']
        self.prediction_window = model_params['n_outputs'] // self.n_features

    def save_model_params(self):
        self.yaml_file = f'{os.path.dirname(self.save_path)}/model_params.yml'
        write_yaml(self.yaml_file, self.model_params)

    def load_model_params(self):
        model_params = read_yaml(self.yaml_file)
        return model_params

    def check_save_path(self):
        files = os.listdir(self.save_path)
        n_files = len([name for name in files if name.split('.')[-1] == 'pth'])
        return n_files > 0

    def create_save_paths(self):
        pass

    def create_model(self):
        pass

    def load_best_model(self):
        pass

    def predict(self, x):
        pass

    def train(self):
        pass

    def inference_on_dataloader(self):
        pass

    def predict_on_series(self):
        pass
    


class RNNSTFT:
    def __init__(self, model_params=None, model_type='single', model=None, save_path='', SEED=23):
        '''
        High-level abstraction class for using NN model on spectral data
        to do time-series prediction

        model_params: [dict] - parameters of NN model
        model: [laser_jitter.model_basic.LSTMForecaster] - model used for prediction
        save_path: [str] - path for model weights and model info to be saved
        '''
        self.model_params = model_params
        self.model_type = model_type
        
        if self.model_type == 'single':
            self.n_models = 1
        elif self.model_type == 'real_imag':
            self.n_models = 2
        elif self.model_type == 'freq_ensemble':
            self.n_models = self.model_params['n_models']
            
        self.device = 'cuda' if model_params['use_cuda'] else 'cpu'
        self.save_path = save_path
        self.save_paths = self.create_save_paths()
       
        # make sure that directory for saving the model exists
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        self.save_model_params()
        
        if model is None:
            self.create_model()
            if self.check_save_path():
                self.load_best_model()
        else:
            self.model = model

        self.SEED = SEED
        self.n_features = model_params['n_features']
        self.prediction_window = model_params['n_outputs'] // self.n_features

    def check_save_path(self):
        files = os.listdir(self.save_path)
        n_files = len([name for name in files if name.split('.')[-1] == 'pth'])
        return n_files > 0

    def create_save_paths(self):
        if self.model_type == 'single':
            self.save_paths = self.save_path + 'model.pth'
        elif self.model_type == 'real_imag':
            self.model = [self.save_path + f'model_{i}.pth' for i in ['real', 'imag']]
        elif self.model_type == 'freq_ensemble':
            self.model = [self.save_path + f'model_{i}.pth' for i in range(self.n_models)]
        return self.save_paths

    def save_model_params(self):
        self.yaml_file = f'{os.path.dirname(self.save_path)}/model_params.yml'
        write_yaml(self.yaml_file, self.model_params)

    def load_model_params(self):
        model_params = read_yaml(self.yaml_file)
        return model_params

    def create_model(self):
        if self.model_type == 'single':
            self.model = LSTMForecaster(**self.model_params).to(self.device)
        elif self.model_type == 'real_imag':
            self.model = [LSTMForecaster(**self.model_params).to(self.device) for i in range(2)]
        elif self.model_type == 'freq_ensemble':
            self.model = [LSTMForecaster(**self.model_params).to(self.device) for i in range(self.n_models)]
        return self.model

    def load_best_model(self):
        if self.model_type == 'single':
            self.model.load_state_dict(torch.load(self.save_paths))
        else:
            for i in range(len(self.paths)):
                self.model[i].load_state_dict(torch.load(self.save_paths[i]))
        return self.model

    def predict(self, x):
        with torch.no_grad():
            prediction = self.model(x).squeeze()
            prediction = prediction.reshape((-1,self.prediction_window,self.n_features))
        return prediction

    def train(self, trainloader, testloader, criterion, optimizer, n_epochs=30, verbose=True):
        if self.model_type == 'single':
            losses = train_model(self.model, trainloader, testloader, criterion, optimizer,
                                 n_epochs=n_epochs, SEED=self.SEED, save_path=self.save_paths,
                                 device=self.device, verbose=True, model_params=self.model_params)
        elif self.model_type == 'real_imag':
            losses = train_model_real_imag(self.model, trainloader, testloader, criterion, optimizer,
                                 n_epochs=n_epochs, SEED=self.SEED, save_path=self.save_paths,
                                 device=self.device, verbose=True, model_params=self.model_params)
        return losses

    def inference_on_long_series(self, series, series_class, sequence_params, dataloader_params,
                                 forecast_window=100):
        '''
        Expect time series as input
        time-series -> stft dataloader -> stft predictions -> time-series prediction
        '''
        nperseg, noverlap = series_class.stft_params['nperseg'], series_class.stft_params['noverlap']
        training_window = sequence_params['training_window']
        prediction_window = sequence_params['prediction_window']
        step = nperseg - noverlap
        start = noverlap + training_window*step
        
        actuals = [series[t0:t0+forecast_window] for t0 in range(start,len(series)-prediction_window)]
        actuals = np.array(actuals)
        stft_series = series_class.transform_series(series)
        stft_loader = create_dataloader(stft_series, sequence_params, dataloader_params,
                                        shuffle=False)
        predictions = []
        for x,y in stft_loader:
            batch_size = x.shape[0]
            stft_prediction = self.predict(x.to(self.device)).cpu().numpy()
            for i in range(batch_size):
                pred = series_class.inverse_transform_series(stft_prediction[i])
                predictions.append(pred[-prediction_window:-prediction_window+forecast_window])
        predictions = np.array(predictions)
        metrics = calculate_metrics(torch.Tensor(predictions.flatten()),
                                    torch.Tensor(actuals.flatten()))
        
        return (predictions, actuals), metrics

    def predict_on_series(self, series, series_class, device='cuda'):
        stft_series = series_class.transform_series(series)
        seq_len = len(stft_series)
        n_features = 1 if series.ndim == 1 else series.shape[1]
        
        x = torch.Tensor(stft_series.reshape((1,seq_len,n_features))).to(device)
        stft_prediction = self.predict(x).cpu().numpy()
        prediction = series_class.inverse_transform_series(stft_prediction.squeeze())
        return prediction
        

    






    

