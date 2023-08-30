'''
Higher-level interface for NN models from model_basic.py including prediction with models,
inference on dataloaders and single time_series
'''
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from laser_jitter.model_basic import LSTMForecaster
from laser_jitter.utils import read_yaml, write_yaml
from laser_jitter.train import train_model
from laser_jitter.inference import predict_on_series

__all__ = ['RNNTemporal']


class RNNTemporal:
    def __init__(self, model_params=None, model=None, save_path='', SEED=23):
        '''

        '''
        self.model_params = model_params
        self.device = 'cuda' if model_params['use_cuda'] else 'cpu'
        self.save_path = save_path
        
        self.model = model if model is not None else self.load_best_model()
        
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
        losses = train_model(self.model, trainloader, testloader, criterion, optimizer,
                             n_epochs=n_epochs, SEED=self.SEED, save_path=self.save_path,
                             device=self.device, verbose=True)
        return losses

    def inference_on_dataloader(self, model, dataloader, dataloader_smooth=None, device='cuda',
                                scaler=None):
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

        # TODO: return scaled predictions and actuals as well???
        metrics = self.calculate_metrics(predictions.flatten(), actuals.flatten())
        if scaler is not None:
            metrics = [scaler.inverse_transform(np.array([metric])[:,None]).squeeze() for metric in metrics]
        
        return (predictions, actuals, actuals_smooth), metrics

    def calculate_metrics(self, predictions, actuals):
        mae = nn.L1Loss(reduction='mean')(predictions, actuals)
        rms = np.sqrt(nn.MSELoss(reduction='mean')(predictions, actuals))
        return (mae, rms)


class RNNSTFT:
    #TODO
    def __init__(self):
        pass






    

