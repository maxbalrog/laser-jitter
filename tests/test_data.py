import numpy as np
from laser_jitter.data import (TimeSeries, TimeSeriesSTFT, TimeSeriesInSTFTOutTime,
                               generate_sequences, generate_sequences_stft_time,
                               SequenceDataset, create_dataloader, 
                               create_dataloader_stft_time)

N = 5000
series = np.random.uniform(size=(N))

N_smooth = 5
smooth_params = {
    'kernel': np.ones(N_smooth)/N_smooth,
}
scaling = 'standard'
train_size = 0.8

window_length = 500
stft_params = {
    'nperseg': window_length,
    'fs': 1e3,
    'boundary': None,
    'noverlap': window_length-1,
    'padded': False,
    'window': 'tukey'
}

filter_params = {
    'thresh_weight': 1,
    'freq_low': 0
}

training_window = 300
prediction_window = 100
sequence_params = {
    'training_window': training_window,
    'prediction_window': prediction_window,
    'step': 1
}

batch_size = 64
dataloader_params = {
    'batch_size': batch_size,
    'drop_last': False,
}

def test_TimeSeries():
    series_class = TimeSeries(series, smooth_params, scaling, train_size)
    N_cut = N - N_smooth//2*2
    N_train = int(train_size * N_cut)
    N_test = N_cut - N_train
    # Test lengths of transformed series
    assert len(series_class.series) == N_cut
    assert len(series_class.train) == N_train
    assert len(series_class.test) == N_test
    assert len(series_class.train_smooth) == len(series_class.train)
    assert len(series_class.test_smooth) == len(series_class.test)
    # Test that initial time-series and back-transformed are the same
    series_, series_smooth = series_class.transform_series(series)
    series_back = series_class.inverse_transform_series(series_, series_class.scaler)
    assert len(series_) == len(series_back)
    assert np.isclose(series[N_smooth//2:N-N_smooth//2], series_back.squeeze()).all()
    # Check shape of dataloaders
    loaders = series_class.create_dataloaders(series_class.train,
                                              series_class.test,
                                              sequence_params, dataloader_params)
    trainloader, testloader = loaders
    for x, y in trainloader:
        assert x.shape == (batch_size,training_window,1)
        break


def test_TimeSeriesSTFT():
    series_class = TimeSeriesSTFT(series, stft_params, scaling, train_size,
                                  filter_params, smooth_params=None)
    N_train = int(train_size * N)
    N_test = N - N_train
    assert len(series_class.series) == N
    assert len(series_class.train) == N_train
    assert len(series_class.test) == N_test
    nw = len(series_class.freq)
    nt_train = N_train - stft_params['noverlap']
    nt_test = N_test - stft_params['noverlap']
    assert len(series_class.t) == nt_train
    assert series_class.train_stft.shape == (nw, nt_train)
    assert series_class.test_stft.shape == (nw, nt_test)
    nw_filt = len(series_class.freq_filt)
    assert series_class.train_stft_filt.shape == (nw, nt_train)
    assert series_class.test_stft_filt.shape == (nw, nt_test)
    assert series_class.train_real.shape == (nt_train, nw_filt)
    assert series_class.train_imag.shape == (nt_train, nw_filt)
    assert series_class.test_real.shape == (nt_test, nw_filt)
    assert series_class.test_imag.shape == (nt_test, nw_filt)
    series_ = series_class.transform_series(series_class.train)
    assert series_.shape == (nt_train, nw_filt*2)
    series_back = series_class.inverse_transform_series(series_)
    assert len(series_back) == N_train

    loaders = series_class.create_dataloaders(series_class.train_real,
                                              series_class.train_imag,
                                              series_class.test_real,
                                              series_class.test_imag,
                                              sequence_params, dataloader_params)
    trainloader, testloader = loaders
    for x, y in trainloader:
        assert x.shape == (batch_size,training_window,nw_filt*2)
        break


def test_generate_sequences():
    for step in [1,10]:
        sequences, targets = generate_sequences(series, training_window,
                                                prediction_window, step=step)
        for i in [0,100]:
            seq_slice = slice(i*step,i*step+training_window)
            tar_slice = slice(i*step+training_window,i*step+training_window+prediction_window)
            assert np.all(sequences[i] == series[seq_slice])
            assert np.all(targets[i] == series[tar_slice])


def test_SequenceDataset():
    step = 1
    sequences, targets = generate_sequences(series, training_window,
                                            prediction_window, step=step)
    dataset = SequenceDataset(sequences, targets)
    assert len(dataset) == len(targets)
    for i in [0,10]:
        seq_slice = slice(i*step,i*step+training_window)
        tar_slice = slice(i*step+training_window,i*step+training_window+prediction_window)
        sequence_torch, target_torch = dataset[i]
        assert np.isclose(sequence_torch.numpy(), series[seq_slice]).all()
        assert np.isclose(target_torch.numpy(), series[tar_slice]).all()


def test_create_dataloader():
    loader = create_dataloader(series, sequence_params, dataloader_params)
    for x,y in loader:
        assert x.shape == (batch_size,training_window)
        assert y.shape == (batch_size,prediction_window)
        break
    
    
    
