'''
Higher-level interface for NN models from model_basic.py including prediction with models,
inference on dataloaders and single time_series
'''
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from laser_jitter.data import create_dataloader, create_dataloader_stft_time
from laser_jitter.model_basic import LSTMForecaster
from laser_jitter.utils import read_yaml, write_yaml
from laser_jitter.train import train_model, train_model_real_imag, train_model_ensemble
from laser_jitter.inference import calculate_metrics

__all__ = ['RNN_abc', 'RNNTemporal', 'RNNSTFT', 'RNNSTFTInTimeOut', 'RNNSTFT_real_imag']


class RNN_abc:
    def __init__(self, model_params=None, model=None, save_folder='', load_model=False, SEED=23):
        '''
        High-level abstraction class for using NN model to do time-series prediction

        model_params: [dict] - parameters of NN model
        model: [laser_jitter.model_basic.LSTMForecaster] - model used for prediction
        save_folder: [str] - path for model weights and model info to be saved
        load_model: [True/False] - whether to load model from `save_folder`
        '''
        self.model_params = model_params
            
        self.device = 'cuda' if model_params['use_cuda'] else 'cpu'
        self.save_folder = save_folder
        self.create_save_path()
       
        # make sure that directory for saving the model exists
        if not load_model:
            Path(os.path.dirname(save_folder)).mkdir(parents=True, exist_ok=True)
            self.save_model_params()
        
        if model is None:
            self.create_model()
            if load_model and self.check_save_folder():
                self.load_best_model()
        else:
            self.model = model

        self.SEED = SEED
        self.n_features = model_params['n_features']
        self.n_out_features = model_params['n_out_features']
        self.prediction_window = model_params['n_outputs'] // self.n_out_features

    def save_model_params(self):
        self.yaml_file = f'{os.path.dirname(self.save_folder)}/model_params.yml'
        write_yaml(self.yaml_file, self.model_params)

    def load_model_params(self):
        model_params = read_yaml(self.yaml_file)
        return model_params

    def check_save_folder(self):
        files = os.listdir(self.save_folder)
        n_files = len([name for name in files if name.split('.')[-1] == 'pth'])
        return n_files > 0

    def create_save_path(self):
        self.save_path = self.save_folder + 'model.pth'

    def create_model(self):
        self.model = LSTMForecaster(**self.model_params).to(self.device)
        return self.model

    def load_best_model(self):
        self.model.load_state_dict(torch.load(self.save_path))
        return self.model

    def predict(self, x):
        pass

    def train(self, trainloader, testloader, criterion, optimizer, n_epochs=30, verbose=True):
        self.model = self.model.to(self.device)
        losses = train_model(self.model, trainloader, testloader, criterion, optimizer,
                             n_epochs=n_epochs, SEED=self.SEED, save_path=self.save_path,
                             device=self.device, verbose=True, model_params=self.model_params,
                             )
        return losses

    def inference_on_dataloader(self):
        pass

    def predict_on_series(self):
        pass


class RNNTemporal(RNN_abc):
    def __init__(self, model_params=None, model=None, save_folder='', load_model=False, SEED=23):
        '''
        High-level abstraction class for using NN model on temporal data
        to do time-series prediction

        model_params: [dict] - parameters of NN model
        model: [laser_jitter.model_basic.LSTMForecaster] - model used for prediction
        save_path: [str] - path for model weights and model info to be saved
        load_model: [True/False] - whether to load model from `save_folder`
        '''
        super().__init__(model_params, model, save_folder, load_model, SEED)

    def predict(self, x):
        with torch.no_grad():
            prediction = self.model(x).squeeze()
        return prediction

    def inference_on_dataloader(self, dataloader, dataloader_smooth=None, device='cuda'):
        self.model = self.model.to(device)
        self.model.eval()
        predictions, actuals, actuals_smooth = [], [], []
        if dataloader_smooth is None:
            for x, y in dataloader:
                p = self.predict(x.to(device)).cpu()
                predictions.append(p)
                actuals.append(y)
        else:
            for (x,y), (x_smooth, y_smooth) in zip(dataloader, dataloader_smooth):
                p = self.predict(x_smooth.to(device)).cpu()
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


class RNNSTFT(RNN_abc):
    def __init__(self, model_params=None, model=None, save_folder='', load_model=False, SEED=23):
        '''
        High-level abstraction class for using NN model on spectral data
        to do time-series prediction

        model_params: [dict] - parameters of NN model
        model: [laser_jitter.model_basic.LSTMForecaster] - model used for prediction
        save_path: [str] - path for model weights and model info to be saved
        load_model: [True/False] - whether to load model from `save_folder`
        '''
        super().__init__(model_params, model, save_folder, load_model, SEED)
        # self.n_out_features = model_params['n_out_features']

    def predict(self, x):
        with torch.no_grad():
            prediction = self.model(x).squeeze()
            prediction = prediction.reshape((-1,self.prediction_window,self.n_out_features))
        return prediction

    def inference_on_dataloader(self, series, series_class, sequence_params, dataloader_params,
                                forecast_window=100):
        '''
        Expect time series as input
        time-series -> stft dataloader -> stft predictions -> time-series prediction
        '''
        nperseg, noverlap = series_class.stft_params['nperseg'], series_class.stft_params['noverlap']
        training_window = sequence_params['training_window']
        prediction_window = sequence_params['prediction_window']
        if series_class.smooth_params is not None:
            N = series_class.smooth_params['kernel'].shape[0]
        else:
            N = 0
        step = nperseg - noverlap
        start = noverlap + training_window*step
        
        actuals = [series[t0:t0+forecast_window] for t0 in range(start+N//2,len(series)-prediction_window-N//2)]
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
        
        x = torch.Tensor(stft_series.reshape((1,seq_len,self.n_features))).to(device)
        stft_prediction = self.predict(x).cpu().numpy()
        prediction = series_class.inverse_transform_series(stft_prediction.squeeze())
        return prediction


class RNNSTFTInTimeOut(RNNSTFT):
    def __init__(self, model_params=None, model=None, save_folder='', load_model=False, SEED=23):
         '''
        High-level abstraction class for using NN model on spectral data
        to do time-series prediction

        model_params: [dict] - parameters of NN model
        model: [laser_jitter.model_basic.LSTMForecaster] - model used for prediction
        save_path: [str] - path for model weights and model info to be saved
        load_model: [True/False] - whether to load model from `save_folder`
        '''
        super().__init__(model_params, model, save_folder, load_model, SEED)

    def inference_on_dataloader(self, series, series_class, sequence_params, dataloader_params,
                                stft_params, forecast_window=100):
        '''
        Expect time series as input
        time-series -> stft dataloader -> time-series prediction
        '''
        nperseg, noverlap = series_class.stft_params['nperseg'], series_class.stft_params['noverlap']
        training_window = sequence_params['training_window']
        prediction_window = sequence_params['prediction_window']
        if series_class.smooth_params is not None:
            N = series_class.smooth_params['kernel'].shape[0]
        else:
            N = 0
        step = nperseg - noverlap
        start = noverlap + training_window*step
        
        actuals = [series[t0:t0+forecast_window] for t0 in range(start+N//2,len(series)-prediction_window-N//2)]
        actuals = np.array(actuals)
        stft_series = series_class.transform_series(series)
        stft_loader = create_dataloader_stft_time(series, stft_series, sequence_params,
                                                  dataloader_params, stft_params, shuffle=False)
        predictions = []
        # predictions, actuals = [], []
        for x,y in stft_loader:
            batch_size = x.shape[0]
            prediction = self.predict(x.to(self.device)).cpu().numpy()
            predictions.append(prediction.flatten())
            # actuals.append(y.flatten())

        # actuals = np.array(actuals)
        predictions = np.concatenate(predictions)
        predictions = series_class.scaler.inverse_transform(predictions[:,None]).squeeze()
        predictions = predictions.reshape(actuals.shape)
        metrics = calculate_metrics(torch.Tensor(predictions.flatten()),
                                    torch.Tensor(actuals.flatten()))
        return (predictions, actuals), metrics
        

class RNNSTFT_real_imag(RNN_abc):
    def __init__(self, model_params=None, model=None, save_folder='', load_model=False, SEED=23):
        '''
        High-level abstraction class for using NN model on spectral data
        to do time-series prediction (two models to predict real/imag part of
        STFT spectrum)

        model_params: [dict] - parameters of NN model
        model: [laser_jitter.model_basic.LSTMForecaster] - model used for prediction
        save_path: [str] - path for model weights and model info to be saved
        load_model: [True/False] - whether to load model from `save_folder`
        '''
        super().__init__(model_params, model, save_folder, load_model, SEED)
        self.n_models = len(self.model)

    def create_save_path(self):
        self.save_path = [self.save_folder + f'model_{i}.pth' for i in ['real', 'imag']]

    def create_model(self):
        self.model = [LSTMForecaster(**self.model_params).to(self.device) for i in range(2)]
        return self.model

    def load_best_model(self):
        for i in range(len(self.model)):
            self.model[i].load_state_dict(torch.load(self.save_path[i]))
        return self.model

    def predict(self, x):
        preds = []
        with torch.no_grad():
            for i in range(self.n_models):
                pred = self.model[i](x[:,:,i*self.n_features:(i+1)*self.n_features]).squeeze()
                pred = pred.reshape((-1,self.prediction_window,self.n_out_features))
                preds.append(pred)
            prediction = torch.cat(preds, dim=-1)
        return prediction

    def train(self, trainloader, testloader, criterion, optimizer, n_epochs=30, verbose=True):
        losses = train_model_real_imag(self.model, trainloader, testloader, criterion, optimizer,
                             n_epochs=n_epochs, SEED=self.SEED, save_path=self.save_path,
                             device=self.device, verbose=True, model_params=self.model_params)
        return losses

    def inference_on_dataloader(self, series, series_class, sequence_params, dataloader_params,
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
        if series_class.smooth_params is not None:
            N = series_class.smooth_params['kernel'].shape[0]
        else:
            N = 0
        
        actuals = [series[t0:t0+forecast_window] for t0 in range(start+N//2,len(series)-prediction_window-N//2)]
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

        # changed from self.n_features * 2
        x = torch.Tensor(stft_series.reshape((1,seq_len,-1))).to(device)
        stft_prediction = self.predict(x).cpu().numpy()
        prediction = series_class.inverse_transform_series(stft_prediction.squeeze())
        return prediction


class RNNSTFT_ensemble(RNNSTFT_real_imag):
    def __init__(self, model_params=None, model=None, save_folder='', load_model=False,
                 n_models=1, SEED=23):
        '''
        High-level abstraction class for using NN model on spectral data
        to do time-series prediction (separate model for each frequency band spectrum)

        model_params: [dict] - parameters of NN model
        model: [laser_jitter.model_basic.LSTMForecaster] - model used for prediction
        save_path: [str] - path for model weights and model info to be saved
        load_model: [True/False] - whether to load model from `save_folder`
        '''
        self.n_models = n_models
        super().__init__(model_params, model, save_folder, load_model, SEED)

    def create_save_path(self):
        self.save_path = [self.save_folder + f'model_{i}.pth' for i in range(self.n_models)]

    def create_model(self):
        self.model = [LSTMForecaster(**self.model_params).to(self.device) for i in range(self.n_models)]
        return self.model

    def load_best_model(self):
        for i in range(self.n_models):
            self.model[i].load_state_dict(torch.load(self.save_path[i]))
        return self.model

    def predict(self, x):
        preds_real, preds_imag = [], []
        with torch.no_grad():
            for i in range(self.n_models):
                pred = self.model[i](x[:,:,i::self.n_models])#.squeeze()
                pred = pred.reshape((-1,self.prediction_window,2))
                preds_real.append(pred[:,:,0][:,:,None])
                preds_imag.append(pred[:,:,1][:,:,None])
            prediction_real = torch.cat(preds_real, dim=2)
            prediction_imag = torch.cat(preds_imag, dim=2)
            prediction = torch.cat([prediction_real, prediction_imag], dim=-1)
        return prediction

    def train(self, trainloader, testloader, criterion, optimizer, n_epochs=30, verbose=True):
        losses = train_model_ensemble(self.model, trainloader, testloader, criterion, optimizer,
                             n_epochs=n_epochs, SEED=self.SEED, save_path=self.save_path,
                             device=self.device, verbose=True, model_params=self.model_params)
        return losses
        

    






    

