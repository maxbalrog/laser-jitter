"""
Functions to use in jitter prediction scripts.
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from laser_jitter.data import TimeSeries
from laser_jitter.model_basic import LSTMForecaster
from laser_jitter.model import RNNTemporal


def get_one_dataset(series, series_params, dataset_params):
    """
    Create datasets from given series and parameters.

    Parameters
    ----------
    series : np.array
        The time series data. The shape should be (n_samples, n_features).
    series_params : dict
        Parameters for the time series. It should contain 'smooth_params', 'train_size'
        and 'scaling'.
    """
    # Create time series class from given series
    series_class = TimeSeries(series, **series_params)

    # Choose regular or smoothed series
    if series_params.get('smooth_params', None):
        train, test = series_class.train, series_class.test
    else:
        train, test = series_class.train_smooth, series_class.test_smooth

    # Create dataloaders
    sequence_params = dataset_params.get('sequence_params', {})
    dataloader_params = dataset_params.get('dataloader_params', {})
    loaders = series_class.create_dataloaders(series_class.train_smooth,
                                              series_class.test_smooth,
                                              sequence_params, dataloader_params)
    trainloader, testloader = loaders
    dataset = {
        "series_class": series_class,
        "trainloader": trainloader,
        "testloader": testloader,
    }
    return dataset


def get_datasets(series, series_params, dataset_params, separate_channels=True):
    """
    Create datasets from given series and parameters.
    """
    if separate_channels:
        datasets = []
        for i in range(series.shape[1]):
            series_i = series[:, i]
            dataset = get_one_dataset(series_i, series_params, dataset_params)
            datasets.append(dataset)
    else:
        datasets = [get_one_dataset(series, series_params, dataset_params)]
    return datasets


def get_default_model_params(n_features, n_lookback, n_forecast):
    use_cuda = torch.cuda.is_available()
    n_hidden_lstm = 64
    model_params = {
        'n_features': n_features,
        'n_hidden_lstm': n_hidden_lstm*n_features,
        'n_hidden_fc': 1000*n_features,
        'n_outputs': n_forecast*n_features,
        'n_out_features': n_features,
        'sequence_len': n_lookback,
        'n_lstm_layers': 2,
        'n_deep_layers': 1,
        'dropout': 0.2,
        'use_cuda': use_cuda
    }
    return model_params


def get_one_model(model_params, save_folder, create_model=True):
    """
    Create a model from given parameters.

    Parameters
    ----------
    model_params : dict
        Parameters for the model.
    save_folder : str
        The folder where the model will be saved.
    """
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    if create_model:
        model = LSTMForecaster(**model_params).to(device)
    else:
        model = None
    model_high_level = RNNTemporal(model_params, model, save_folder)
    return model_high_level


def get_models(model_params, save_folder, create_model=True, separate_channels=True,
               n_channels=2):
    """
    Create models from given parameters.

    Parameters
    ----------
    model_params : dict
        Parameters for the model.
    save_folder : str
        The folder where the model will be saved.
    """
    models = []
    if separate_channels:
        for i in range(n_channels):
            models.append(get_one_model(model_params, os.path.join(save_folder, f"model_{i}"),
                                        create_model))
    else:
        models.append(get_one_model(model_params, save_folder, create_model))
    return models


def plot_losses(losses, save_folder):
    plt.figure(figsize=(16,6))
    for i,title in enumerate(['Train', 'Test']):
        plt.subplot(1,2,i+1)
        plt.plot(losses[i])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.grid()
        if i == 1:
            idx = np.argmin(losses[1])
            plt.plot(idx, losses[1][idx], '*', color='red', ms=20)
    plt.savefig(os.path.join(os.path.dirname(save_folder), 'losses.png'),
                bbox_inches='tight')
    plt.show()


def train_model(model, dataset, save_folder, n_epochs=10, lr=1e-4,
                verbose=False):
    """
    Train the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    trainloader : torch.utils.data.DataLoader
        The training data loader.
    testloader : torch.utils.data.DataLoader
        The testing data loader.
    model_params : dict
        Parameters for the model.
    save_folder : str
        The folder where the model will be saved.
    """
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    trainloader, testloader = dataset["trainloader"], dataset["testloader"]

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr)

    t_start = time.perf_counter()
    losses = model.train(trainloader, testloader, criterion, optimizer,
                         n_epochs=n_epochs)
    t_end = time.perf_counter()
    print("="*30)
    print(f"Training time: {t_end - t_start:.2f} seconds")

    if verbose:
        plot_losses(losses, save_folder)

    print(f"Model is trained. Saving the best model to {save_folder}.")


def train_models(models, datasets, save_folder, n_epochs=10, lr=1e-4,
                 verbose=False, separate_channels=True):
    for i,(model,dataset) in enumerate(zip(models, datasets)):
        save_local = os.path.join(save_folder, f"model_{i}") if separate_channels else save_folder
        train_model(model, dataset, save_local, n_epochs=n_epochs, lr=lr,
                    verbose=verbose)
    print("All models are trained...")


def predict_on_series(series, models, datasets, device='cuda'):
    predictions = []
    for i,(model,dataset) in enumerate(zip(models,datasets)):
        series_local = series[:,i] if len(models) > 1 else series
        prediction = model.predict_on_series(series_local, dataset['series_class'],
                                             device=device)
        predictions.append(prediction)
    return np.vstack(predictions).T
