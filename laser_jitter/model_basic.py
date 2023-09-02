'''
Basic NN architectures used for time-series forecasting. Currently only LSTM
followed by FC layers is supported.
'''
import torch
import torch.nn as nn

__all__ = ['LSTMForecaster']


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, n_hidden_lstm, n_hidden_fc, n_outputs, sequence_len,
                 n_lstm_layers=1, n_deep_layers=10, dropout=0.2, use_cuda=False):
        '''
        n_features: [int] - number of input features (1 for univariate forecasting)
        n_hidden_lstm: [int] - number of neurons in each hidden layer of LSTM
        n_hidden_fc: [int] - number of neurons in each hidden layer of FC layers following LSTM
        n_outputs: [int] - number of outputs to predict for each training example
        n_deep_layers: [int] - number of hidden dense layers after the lstm layer
        sequence_len: [int] - number of steps to look back at for prediction
        dropout: [float (0 < dropout < 1)] - dropout ratio between dense layers
        '''
        super().__init__()

        self.n_lstm_layers = n_lstm_layers
        self.nhid_lstm = n_hidden_lstm
        self.nhid_fc = n_hidden_fc
        self.use_cuda = use_cuda # set option for device selection
        self.device = 'cuda' if use_cuda else 'cpu'

        # LSTM Layer
        self.lstm = nn.LSTM(n_features,
                            n_hidden_lstm,
                            num_layers=n_lstm_layers,
                            batch_first=True) # As we have transformed our data in this way

        # first dense after lstm
        self.fc1 = nn.Linear(n_hidden_lstm * sequence_len, n_hidden_fc) 
        # Dropout layer 
        self.dropout = nn.Dropout(p=dropout)

        # Create fully connected layers (n_hidden x n_deep_layers)
        dnn_layers = []
        for i in range(n_deep_layers):
            # Last layer (n_hidden x n_outputs)
            if i == n_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(self.nhid_fc, n_outputs))
            # All other layers (n_hidden x n_hidden) with dropout option
            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(self.nhid_fc, self.nhid_fc))
                if dropout:
                    dnn_layers.append(nn.Dropout(p=dropout))
            # compile DNN layers
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):
        x, h = self.lstm(x)
        x = self.dropout(x.contiguous().view(x.shape[0], -1)) # Flatten lstm out
        x = self.fc1(x) # First Dense
        return self.dnn(x) # Pass forward through fully connected DNN.