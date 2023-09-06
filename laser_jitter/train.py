'''
Generic training protocols for a single model, two models operating on real and imaginary part 
of the spectrum and model ensemble
'''
import os
from pathlib import Path
import numpy as np
import torch
from laser_jitter.utils import fix_seed

__all__ = ['train_model', 'train_model_real_imag', 'train_model_ensemble']

    
def train_model(model, trainloader, testloader, criterion, optimizer,
                n_epochs=50, SEED=23, save_path='models/rnn.pth', device='cpu', verbose=True,
                model_params=None):
    '''
    model: [torch.nn.Module] - NN model
    trainloader: [torch.utils.data.DataLoader] - dataloader of train data
    testloader: [torch.utils.data.DataLoader] - dataloader of test data
    criterion: [torch metric] - metric to access the performance of the model
    optimizer: [torch optimizer] - optimizer of model parameters
    n_epochs: [int] - number of training epochs
    save_path: [str] - path to save model to
    device: ['cpu' or 'cuda'] - device to perform calculations
    verbose: [True or False] - whether to print epoch loss information
    model_params: [dict] - parameters of model
    '''
    fix_seed(SEED)

    n_features = model_params['n_out_features']
    prediction_window = model_params['n_outputs'] // n_features
    
    t_losses, v_losses = [], []
    best_valid_loss = 1e5
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0.0, 0.0

        # train step
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            # Forward Pass
            preds = model(x).squeeze()
            preds = preds.reshape((-1,prediction_window,n_features))
            # preds = model.predict(x)
            loss = criterion(preds, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss_mean = train_loss / len(trainloader)
        t_losses.append(train_loss_mean)

        # validation step
        model.eval()
        for x, y in testloader:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                preds = model(x).squeeze()
                preds = preds.reshape((-1,prediction_window,n_features))
                # preds = model.predict(x)
                loss = criterion(preds, y)
            valid_loss += loss.item()
        valid_loss_mean = valid_loss / len(testloader)
        v_losses.append(valid_loss_mean)
        if valid_loss_mean < best_valid_loss:
            best_valid_loss = valid_loss_mean
            torch.save(model.state_dict(), save_path)

        if verbose:
            print(f'{epoch} - train: {train_loss_mean}, valid: {valid_loss_mean}')
    return t_losses, v_losses


def train_model_real_imag(models, trainloader, testloader, criterion, optimizers,
                n_epochs=50, SEED=23, save_path=None, device='cpu', verbose=True,
                model_params=None):
    '''
    models: [list of torch.nn.Module] - NN models
    trainloader: [torch.utils.data.DataLoader] - dataloader of train data
    testloader: [torch.utils.data.DataLoader] - dataloader of test data
    criterion: [torch metric] - metric to access the performance of the model
    optimizers: [list of torch optimizers] - optimizers of model parameters
    n_epochs: [int] - number of training epochs
    save_path: [str] - path to save model to
    device: ['cpu' or 'cuda'] - device to perform calculations
    verbose: [True or False] - whether to print epoch loss information
    model_params: [dict] - parameters of model
    '''
    fix_seed(SEED)

    n_models = len(models)
    n_features = model_params['n_features']
    prediction_window = model_params['n_outputs'] // n_features
    
    t_losses, v_losses = [], []
    best_valid_loss = [1e5, 1e5]
    for epoch in range(n_epochs):
        train_loss, valid_loss = [np.zeros(n_models) for i in range(2)]

        # train step
        for i in range(n_models):
            models[i].train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            for i in range(n_models):
                preds = models[i](x[:,:,i*n_features:(i+1)*n_features]).squeeze()
                preds = preds.reshape((-1,prediction_window,n_features))
                loss = criterion(preds, y[:,:,i*n_features:(i+1)*n_features])
                train_loss[i] += loss.item()
                loss.backward()
                optimizers[i].step()
                optimizers[i].zero_grad()
        train_loss_mean = train_loss / len(trainloader)
        t_losses.append(train_loss_mean)

        # validation step
        for i in range(n_models):
            models[i].eval()
        for x, y in testloader:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                for i in range(n_models):
                    preds = models[i](x[:,:,i*n_features:(i+1)*n_features]).squeeze()
                    preds = preds.reshape((-1,prediction_window,n_features))
                    loss = criterion(preds, y[:,:,i*n_features:(i+1)*n_features])
                    valid_loss[i] += loss.item()
        valid_loss_mean = valid_loss / len(testloader)
        v_losses.append(valid_loss_mean)
        for i in range(n_models):
            if valid_loss_mean[i] < best_valid_loss[i]:
                best_valid_loss[i] = valid_loss_mean[i]
                torch.save(models[i].state_dict(), save_path[i])

        if verbose:
            print(f'{epoch} - train: {train_loss_mean.mean()}, valid: {valid_loss_mean.mean()}')
    return t_losses, v_losses


def train_model_ensemble(models, trainloader, testloader, criterion, optimizers,
                n_epochs=50, SEED=23, save_path=None, device='cpu', verbose=True,
                model_params=None):
    '''
    models: [list of torch.nn.Module] - NN models
    trainloader: [torch.utils.data.DataLoader] - dataloader of train data
    testloader: [torch.utils.data.DataLoader] - dataloader of test data
    criterion: [torch metric] - metric to access the performance of the model
    optimizers: [list of torch optimizers] - optimizers of model parameters
    n_epochs: [int] - number of training epochs
    save_path: [str] - path to save model to
    device: ['cpu' or 'cuda'] - device to perform calculations
    verbose: [True or False] - whether to print epoch loss information
    model_params: [dict] - parameters of model
    '''
    fix_seed(SEED)

    n_models = len(models)
    n_features = model_params['n_features']
    prediction_window = model_params['n_outputs'] // n_features
    
    t_losses, v_losses = [], []
    best_valid_loss = [1e5 for i in range(n_models)]
    for epoch in range(n_epochs):
        train_loss, valid_loss = [np.zeros(n_models) for i in range(2)]

        # train step
        for i in range(n_models):
            models[i].train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            for i in range(n_models):
                preds = models[i](x[:,:,i::n_models]).squeeze()
                preds = preds.reshape((-1,prediction_window,n_features))
                loss = criterion(preds, y[:,:,i::n_models])
                train_loss[i] += loss.item()
                loss.backward()
                optimizers[i].step()
                optimizers[i].zero_grad()
        train_loss_mean = train_loss / len(trainloader)
        t_losses.append(train_loss_mean)

        # validation step
        for i in range(n_models):
            models[i].eval()
        for x, y in testloader:
            with torch.no_grad():
                x, y = x.to(device), y.to(device)
                for i in range(n_models):
                    preds = models[i](x[:,:,i::n_models]).squeeze()
                    preds = preds.reshape((-1,prediction_window,n_features))
                    loss = criterion(preds, y[:,:,i::n_models])
                    valid_loss[i] += loss.item()
        valid_loss_mean = valid_loss / len(testloader)
        v_losses.append(valid_loss_mean)
        for i in range(n_models):
            if valid_loss_mean[i] < best_valid_loss[i]:
                best_valid_loss[i] = valid_loss_mean[i]
                torch.save(models[i].state_dict(), save_path[i])

        if verbose:
            print(f'{epoch} - train: {train_loss_mean.mean()}, valid: {valid_loss_mean.mean()}')
    return t_losses, v_losses
