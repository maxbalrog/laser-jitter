'''
Generic training protocol
'''
import os
from pathlib import Path
import torch
from laser_jitter.utils import fix_seed

__all__ = ['train_model']

    
def train_model(model, trainloader, testloader, criterion, optimizer,
                n_epochs=50, SEED=23, save_path='models/rnn.pth', device='cpu', verbose=True):
    fix_seed(SEED)

    t_losses, v_losses = [], []
    best_valid_loss = 1e5
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0.0, 0.0

        # train step
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.squeeze().to(device)
            # Forward Pass
            preds = model(x).squeeze()
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
                x, y = x.to(device), y.squeeze().to(device)
                preds = model(x).squeeze()
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