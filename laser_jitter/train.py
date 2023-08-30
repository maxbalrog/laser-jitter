'''
Generic training protocol
'''
import os
from pathlib import Path
import torch
from laser_jitter.utils import fix_seed


# def train_step(model, trainloader, optimizer, criterion, device='cuda'):
#     train_loss = 0.0
#     model.train()
#     for x, y in trainloader:
#         optimizer.zero_grad()
#         x, y = x.to(device), y.squeeze().to(device) # removed .squeeze()
#         preds = model(x)#.squeeze()
#         loss = criterion(preds, y) # compute batch loss
#         train_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     train_loss_mean = train_loss / len(trainloader)
#     return train_loss_mean


# def valid_step(model, validloader, criterion, device='cuda'):
#     valid_loss = 0.0
#     model.eval()
#     for x, y in validloader:
#         with torch.no_grad():
#             x, y = x.to(device), y.squeeze().to(device)
#             preds = model(x)#.squeeze()
#             loss = criterion(preds, y)
#         valid_loss += loss.item()
#     valid_loss_mean = valid_loss / len(validloader)
#     return valid_loss_mean


# def train_model_(model, trainloader, testloader, criterion, optimizer, train_step, valid_step,
#                 n_epochs=50, SEED=23, save_path='models/rnn.pth', device='cpu', verbose=True):
#     fix_seed(SEED)
#     # make sure that directory for saving the model exists
#     Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

#     t_losses, v_losses = [], []
#     best_valid_loss = 1e5
#     for epoch in range(n_epochs):
#         # train step
#         train_loss = train_step(model, trainloader, optimizer, criterion, device)
#         t_losses.append(train_loss)

#         # validation step
#         valid_loss = valid_step(model, trainloader, criterion, device)
#         v_losses.append(valid_loss)
        
#         if valid_loss < best_valid_loss:
#             best_valid_loss = valid_loss
#             torch.save(model.state_dict(), save_path)

#         if verbose:
#             print(f'{epoch} - train: {train_loss}, valid: {valid_loss}')
#     return t_losses, v_losses

    
def train_model(model, trainloader, testloader, criterion, optimizer,
                n_epochs=50, SEED=23, save_path='models/rnn.pth', device='cpu', verbose=True):
    fix_seed(SEED)
    # make sure that directory for saving the model exists
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    t_losses, v_losses = [], []
    best_valid_loss = 1e5
    for epoch in range(n_epochs):
        train_loss, valid_loss = 0.0, 0.0

        # train step
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.squeeze().to(device)
            # Forward Pass
            # preds = model(x).squeeze()
            preds = model.predict(x)
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
                # preds = model(x).squeeze()
                preds = model.predict(x)
                loss = criterion(preds, y)
            valid_loss += loss.item()
        valid_loss_mean = valid_loss / len(testloader)
        v_losses.append(valid_loss_mean)
        if valid_loss_mean < best_valid_loss:
            best_valid_loss = valid_loss_mean
            torch.save(model.state_dict(), save_path)

        if verbose:
            print(f'{epoch} - train: {train_loss}, valid: {valid_loss}')
    return t_losses, v_losses