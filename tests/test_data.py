import numpy as np
from laser_jitter.data import (TimeSeries, TimeSeriesSTFT, TimeSeriesInSTFTOutTime,
                               generate_sequences, generate_sequences_stft_time,
                               SequenceDataset, create_dataloader, create_dataloader_stft_time)

N = 1000
series = np.random.uniform(size=(N))

N_smooth = 5
smooth_params = {
    'kernel': np.ones(N_smooth)/N_smooth,
}
scaling = 'standard'
train_size = 0.8

training_window = 300
prediction_window = 100
sequence_params = {
    'training_window': training_window,
    'prediction_window': prediction_window,
    'step': 1
}

batch_size = 64
dataloader_params = {
    'batch_size': batch_sze,
    'drop_last': False,
}

def test_TimeSeries():
    series_class = TimeSeries(series, smooth_params, scaling, train_size)
    assert len(series_class.series) == N - N_smooth//2*2
    assert len(series_class.train) == int(train_size * N)
    assert len(series_class.test) == int((1-train_size) * N)
    assert len(series_class.train_smooth) == len(series_class.train)
    assert len(series_class.test_smooth) == len(series_class.test)
    series_, series_smooth = series_class.transform_series(series)
    series_back = series_class.inverse_transform_series(series_smooth)
    assert len(series_) == len(series_back)
    assert np.isclose(series[N_smooth//2:N-N_smooth//2], series_back)
    loaders = series_class.create_dataloaders(series_class.train,
                                              series_class.test,
                                              sequence_params, dataloader_params)
    trainloader, testloader = loaders
    for x, y in trainloader:
        assert x.shape == (batch_size,training_window,1)
        break


def test_TimeSeriesSTFT():
    pass
    
    
