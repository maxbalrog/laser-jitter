'''
This script provides data classes for time series analysis.
'''

import numpy as np
from scipy.signal import stft, istft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ['TimeSeries', 'TimeSeriesSTFT', 'generate_sequences', 'SequenceDataset',
           'create_dataloader']


class TimeSeries:
    def __init__(self, series, smooth_params=None, scaling='standard', train_size=0.8):
        '''
        This class provides necessary tools to prepare a single time-series for NN model
        working with temporal features.
        
        series: [numpy ndarray] - univariate time-series
        smooth_params: [dict] - parameters for smoothing, e.g. {'kernel': np.ones(5)/5}
        scaling: [str] - scaling to be used, one of ['standard', 'minmax']
        train_size: [float] - fraction of training data
        '''
        self.series = series
        self.smooth_params = smooth_params
        assert scaling in ['standard', 'minmax'], f"Only ['standard', 'minmax'] scalings are supported \
        but you passed {scaling}"
        self.scaling = scaling
        self.train_size = train_size

        self.smooth_split_and_scale()

    def smooth(self, series):
        series_smooth = np.convolve(series, self.smooth_params['kernel'], mode='valid')
        return series_smooth

    def train_test_split(self, series):
        train, test = train_test_split(series, train_size=self.train_size, shuffle=False)
        return train, test

    def scale(self, series, scaler=None):
        if series.ndim == 1:
            series = series[:,np.newaxis]
        if scaler is None:
            scaler = StandardScaler() if self.scaling == 'standard' else MinMaxScaler((-1,1))
            scaler.fit(series)
        series_scaled = scaler.transform(series)
        return series_scaled, scaler

    def split_and_scale(self, series):
        train, test = self.train_test_split(series)
        train, scaler = self.scale(train)
        test, _ = self.scale(test, scaler)
        data = {
            'train': train,
            'test': test,
            'scaler': scaler
        }
        return data

    def smooth_split_and_scale(self):
        # TODO: Do we actually need these results variable???
        results = {}
        if self.smooth_params is not None:
            self.series_smooth = self.smooth(self.series)
            data_smooth = self.split_and_scale(self.series_smooth)
            self.train_smooth, self.test_smooth, self.scaler_smooth = data_smooth.values()
            results['smoothed'] = data_smooth

            # Cut out smooth boundary region from unsmoothed timeseries
            # (useful for metric computation and comparison between smoothed and non-smoothed cases)
            N = len(self.smooth_params['kernel'])
            self.series = self.series[N//2:len(self.series)-N//2]

        data = self.split_and_scale(self.series)
        self.train, self.test, self.scaler = data.values()
        results['unchanged'] = data
        return results

    @staticmethod
    def create_dataloaders(train, test, sequence_params, dataloader_params):
        trainloader = create_dataloader(train, sequence_params, dataloader_params)
        testloader = create_dataloader(test, sequence_params, dataloader_params)
        return trainloader, testloader


class TimeSeriesSTFT:
    def __init__(self, series, stft_params, scaling='standard', train_size=0.8, filter_params):
        '''
        This class provides necessary tools to prepare a single time-series for NN
        working with STFT features

        
        '''
        self.series = series
        self.stft_params = stft_params
        self.istft_params = {key: val for key,val in stft_params.items() if key != 'padded'}
        assert scaling in ['standard', 'minmax'], f"Only ['standard', 'minmax'] scalings are supported \
        but you passed {scaling}"
        self.scaling = scaling
        self.train_size = train_size
        self.filter_params = filter_params

        self.split_stft_filter_scale(self.filter_params)

    def calculate_stft(self, series):
        freq, t, spectrum = stft(series, **self.stft_kwargs)
        return freq, t, spectrum

    def filter_stft(self, freq, spectrum, filter_params, idx_filt=None):
        # filt stands for filtered
        spectrum_filt = spectrum.copy()
        amplitude = np.abs(spectrum).sum(axis=1)
        
        if idx_filt is None:
            amplitude = np.abs(spectrum).sum(axis=1)
            thresh = filter_params['thresh_weight'] * np.mean(amplitude)
            idx = amplitude < thresh
            idx_freq = freq < filter_params['freq_low']
            idx[idx_freq] = True
            idx_filt = np.logical_not(idx)
        freq_filt = freq[idx_filt]
        spectrum_filt[idx] = 0
        return idx_filt, freq_filt, spectrum_filt  

    def train_test_split(self, series):
        train, test = train_test_split(series, train_size=self.train_size, shuffle=False)
        return train, test

    def scale(self, series, scaler=None):
        if scaler is None:
            scaler = StandardScaler() if self.scaling == 'standard' else MinMaxScaler((-1,1))
            scaler.fit(series)
        series_scaled = scaler.transform(series)
        return series_scaled, scaler

    def split_stft_filter_scale(self, filter_params, combine_real_imag=True):
        self.train, self.test = self.train_test_split(self.series)

        # calculate and filter stft
        self.freq, self.t, self.train_stft = self.calculate_stft(self.train)
        self.idx_filt, self.freq_filt, self.train_stft_filt = self.filter_stft(self.freq, self.train_stft,
                                                                               filter_params)
        self.train_real, self.train_imag = np.real(self.train_stft_filt), np.imag(self.train_stft_filt)
        if combine_real_imag:
            self.train_real_imag = np.vstack([self.train_real, self.train_imag])
            self.train_real_imag, self.scaler_real_imag = self.scale(self.train_real_imag)
        else:
            self.train_real, self.scaler_real = self.scale(self.train_real)
            self.train_imag, self.scaler_imag = self.scale(self.train_imag)
        
        _, _, self.test_stft = self.calculate_stft(self.test)
        _, _, self.test_stft_filt = self.filter_stft(self.freq, self.test_stft, filter_params, self.idx_filt)
        self.test_real, self.test_imag = np.real(self.test_stft_filt), np.imag(self.test_stft_filt)
        self.test_real, _ = self.scale(self.test_real, self.scaler_real)
        self.test_imag, _ = self.scale(self.test_imag, self.scaler_imag)

    #TODO: implement option of combining real and imaginary parts into one feature array
    #TODO: add possibility of creating dataloaders for single frequency bands

    @staticmethod
    def create_dataloaders(train, test, sequence_params, dataloader_params):
        trainloader = create_dataloader(train, sequence_params, dataloader_params)
        testloader = create_dataloader(test, sequence_params, dataloader_params)
        return trainloader, testloader

    

def generate_sequences(series, training_window, prediction_window, step=1):
    '''
    series: Time Series [numpy ndarray (series_len, n_features)] - time-series
    training_window [int] - how many steps to look back
    prediction_window [int] - how many steps forward to predict
    step: [int] - step between two sequences in data, e.g. (step=1, tw=5) results in sequences [0:5], [1:6], ...
          (step=3, tw=5) -> [0:5], [3:8], ...

    returns: dictionary of sequences and targets for all sequences
    '''
    L = len(series)
    tw, pw = training_window, prediction_window
    sequences, targets = [], []
    for i in range(0, L-tw-pw, step):
        # Get current sequence 
        sequences.append(series[i:i+tw])
        # Get values right after the current sequence
        targets.append(series[i+tw:i+tw+pw])
    return sequences, targets


class SequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __getitem__(self, idx):
        return torch.Tensor(self.sequences[idx]), torch.Tensor(self.targets[idx])
        # return self.sequences[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
    

def create_dataloader(series, sequence_params, dataloader_params):
    '''
    series: [np.ndarray] - time-series
    sequence_params: [dict] - {'training_window': 200, 'prediction_window': 100, 'step': 1}
    dataloader_params: [dict] - {'batch_size': 64, 'shuffle': True, 'drop_last': False}
    '''
    sequences, targets = generate_sequences(series, **sequence_params)
    dataset = SequenceDataset(sequences, targets)
    loader = DataLoader(dataset, **dataloader_params)
    return loader
    
