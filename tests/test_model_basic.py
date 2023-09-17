import torch 
from laser_jitter.model_basic import LSTMForecaster

n_features = 1
n_hidden_lstm = 64
training_window = 300
prediction_window = 100

model_params = {
    'n_features': n_features,
    'n_hidden_lstm': n_hidden_lstm,
    'n_hidden_fc': 100,
    'n_outputs': prediction_window*n_features,
    'n_out_features': 1,
    'sequence_len': training_window,
    'n_lstm_layers': 2,
    'n_deep_layers': 1,
    'dropout': 0.2,
    'use_cuda': False
}

batch_size = 32
x = torch.rand((batch_size,training_window,n_features))

def test_LSTMForecaster():
    model = LSTMForecaster(**model_params)
    out = model(x)
    assert out.shape == (batch_size,prediction_window*n_features)

