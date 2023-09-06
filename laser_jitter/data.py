'''
Data classes for time series analysis and functions for creating dataloaders
'''

import numpy as np
from scipy.signal import stft, istft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ['TimeSeries', 'TimeSeriesSTFT', 'TimeSeriesInSTFTOutTime',
           'generate_sequences', 'generate_sequences_stft_time', 'SequenceDataset',
           'create_dataloader', 'create_dataloader_stft_time']


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
        trainloader = create_dataloader(train, sequence_params, dataloader_params,
                                        shuffle=True)
        testloader = create_dataloader(test, sequence_params, dataloader_params,
                                       shuffle=False)
        return trainloader, testloader

    def transform_series(self, series):
        if self.smooth_params is not None:
            series_smooth = self.smooth(series)
            series_smooth = self.scaler_smooth.transform(series_smooth[:, np.newaxis])
            N = len(self.smooth_params['kernel'])
            series = series[N//2:len(series)-N//2]
        series = self.scaler.transform(series[:, np.newaxis])
        return series, series_smooth

    def inverse_transform_series(self, series, scaler=None):
        if series.ndim == 1:
            series = series[:,np.newaxis]
        if scaler is None:
            series_scaled = self.scaler_smooth.inverse_transform(series).squeeze()
        else:
            series_scaled = scaler.inverse_transform(series).squeeze()
        return series_scaled


class TimeSeriesSTFT:
    def __init__(self, series, stft_params, scaling='standard', train_size=0.8, filter_params=None,
                 smooth_params=None):
        '''
        This class provides necessary tools to prepare a single time-series for NN
        working with STFT features

        series: [numpy ndarray] - univariate time-series
        stft_params: [dict] - parameters for STFT (see scipy documentation)
        scaling: [str] - scaling to be used, one of ['standard', 'minmax']
        train_size: [float] - fraction of training data
        filter_params: [dict] - parameters to filter non-relevant frequencies from STFT spectrum,
                                e.g., filter_params = {'thresh_weight': 1, 'freq_low': 0} where
                                frequency threshold is determined by 
                                'thresh_weight' * np.mean(np.abs(stft).sum(axis=1))
                                and all frequencies lower than 'freq_low' are also filtered out
        smooth_params: [dict] - parameters for smoothing
        '''
        self.series = series
        self.stft_params = stft_params
        self.istft_params = {key: val for key,val in stft_params.items() if key != 'padded'}
        assert scaling in ['standard', 'minmax'], f"Only ['standard', 'minmax'] scalings are supported \
        but you passed {scaling}"
        self.scaling = scaling
        self.train_size = train_size
        self.filter_params = filter_params
        self.smooth_params = smooth_params
        if smooth_params is not None:
            self.do_smooth()

        self.split_stft_filter_scale(self.filter_params)

    def smooth(self, series):
        series_smooth = np.convolve(series, self.smooth_params['kernel'], mode='valid')
        return series_smooth

    def do_smooth(self):
        self.series_smooth = self.smooth(self.series)
        N = len(self.smooth_params['kernel'])
        self.series = self.series[N//2:len(self.series)-N//2]

    def calculate_stft(self, series):
        freq, t, spectrum = stft(series, **self.stft_params)
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
        spectrum_filt[np.logical_not(idx_filt)] = 0
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

    def split_stft_filter_scale(self, filter_params):
        if self.smooth_params is None:
            self.train, self.test = self.train_test_split(self.series)
        else:
            self.train, self.test = self.train_test_split(self.series_smooth)

        # calculate and filter stft
        self.freq, self.t, self.train_stft = self.calculate_stft(self.train)
        self.idx_filt, self.freq_filt, self.train_stft_filt = self.filter_stft(self.freq, self.train_stft,
                                                                               filter_params)
        self.train_real = np.real(self.train_stft_filt[self.idx_filt])
        self.train_imag = np.imag(self.train_stft_filt[self.idx_filt])
        self.train_real, self.scaler_real = self.scale(self.train_real.T)
        self.train_imag, self.scaler_imag = self.scale(self.train_imag.T)
        
        _, _, self.test_stft = self.calculate_stft(self.test)
        _, _, self.test_stft_filt = self.filter_stft(self.freq, self.test_stft, filter_params, self.idx_filt)
        self.test_real = np.real(self.test_stft_filt[self.idx_filt])
        self.test_imag = np.imag(self.test_stft_filt[self.idx_filt])
        self.test_real, _ = self.scale(self.test_real.T, self.scaler_real)
        self.test_imag, _ = self.scale(self.test_imag.T, self.scaler_imag)
    
    @staticmethod
    def create_dataloaders(train_real, train_imag, test_real, test_imag, sequence_params, dataloader_params):
        train = np.hstack([train_real, train_imag])
        test = np.hstack([test_real, test_imag])
        trainloader = create_dataloader(train, sequence_params, dataloader_params, shuffle=True)
        testloader = create_dataloader(test, sequence_params, dataloader_params, shuffle=False)
        return trainloader, testloader

    def transform_series(self, series):
        if self.smooth_params is not None:
            series = self.smooth(series)
        _, _, stft_spectrum = self.calculate_stft(series)
        _, _, stft_spectrum_filt = self.filter_stft(self.freq, stft_spectrum, self.filter_params, self.idx_filt)
        real = np.real(stft_spectrum_filt[self.idx_filt])
        imag = np.imag(stft_spectrum_filt[self.idx_filt])
        
        real, _ = self.scale(real.T, self.scaler_real)
        imag, _ = self.scale(imag.T, self.scaler_imag)
        return np.hstack([real, imag])

    def inverse_transform_series(self, series):
        '''
        Expecting series (prediction_window, n_freq_filt*2) where first half of features
        correspond to real part and second half to imaginary
        '''
        n_freq_filt, n_freq = len(self.freq_filt), len(self.freq)
        real = self.scaler_real.inverse_transform(series[:,:n_freq_filt]).T
        imag = self.scaler_imag.inverse_transform(series[:,n_freq_filt:]).T
        stft_spectrum = np.zeros((len(self.freq),series.shape[0]), dtype=np.complex128)
        # print(stft_spectrum.shape, real.shape, imag.shape)
        idx = np.arange(n_freq)[self.idx_filt]
        for i,j in enumerate(idx):
            stft_spectrum[j] = real[i] + 1j*imag[i]
        t, series = istft(stft_spectrum, **self.istft_params)
        return series


class TimeSeriesInSTFTOutTime(TimeSeriesSTFT):
    def __init__(self, series, stft_params, scaling='standard', train_size=0.8, filter_params=None,
                 smooth_params=None):
        '''
        This class provides necessary tools to prepare a single time-series for NN
        taking STFT features as input and producing temporal features as output

        series: [numpy ndarray] - univariate time-series
        stft_params: [dict] - parameters for STFT (see scipy documentation)
        scaling: [str] - scaling to be used, one of ['standard', 'minmax']
        train_size: [float] - fraction of training data
        filter_params: [dict] - parameters to filter non-relevant frequencies from STFT spectrum,
                                e.g., filter_params = {'thresh_weight': 1, 'freq_low': 0} where
                                frequency threshold is determined by 
                                'thresh_weight' * np.mean(np.abs(stft).sum(axis=1))
                                and all frequencies lower than 'freq_low' are also filtered out
        smooth_params: [dict] - parameters for smoothing
        '''
        super().__init__(series, stft_params, scaling='standard', train_size=0.8,
                         filter_params=filter_params, smooth_params=smooth_params)
        self.scale_series()

    def scale_series(self):
        self.train, self.scaler = self.scale(self.train[:,np.newaxis])
        self.test, _ = self.scale(self.test[:,np.newaxis], self.scaler)

    def create_dataloaders(self, train_real, train_imag, train_series, test_real, test_imag,
                           test_series, sequence_params, dataloader_params, stft_params):
        train_stft = np.hstack([train_real, train_imag])
        test_stft = np.hstack([test_real, test_imag])
        trainloader = create_dataloader_stft_time(train_series, train_stft, sequence_params,
                                        dataloader_params, stft_params, shuffle=True)
        testloader = create_dataloader_stft_time(test_series, test_stft, sequence_params,
                                       dataloader_params, stft_params, shuffle=False)
        return trainloader, testloader

    def inverse_transform_series(self, series):
        if series.ndim == 1:
            series = series[:,np.newaxis]
        series = self.scaler.inverse_transform(series)
        return series


def generate_sequences(series, training_window, prediction_window, step=1):
    '''
    series: Time Series [numpy ndarray (series_len, n_features)] - time-series
    training_window [int] - how many steps to look back
    prediction_window [int] - how many steps forward to predict
    step: [int] - step between two sequences in data, e.g. (step=1, tw=5) results 
    in sequences [0:5], [1:6], ...; (step=3, tw=5) -> [0:5], [3:8], ...

    returns: sequences and targets for all sequences
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


def generate_sequences_stft_time(series, series_stft, training_window, prediction_window,
                                 stft_params, step=1):
    '''
    series: Time Series [numpy ndarray (series_len, n_features)] - time-series
    series_stft: STFT Time Series [numpy ndarray (series_len, n_features)] - time-series
                 of STFT features 
    training_window [int] - how many steps to look back
    prediction_window [int] - how many steps forward to predict
    stft_params: [dict] - parameters for STFT (see scipy documentation)
    step: [int] - step between two sequences in data, e.g. (step=1, tw=5) results 
    in sequences [0:5], [1:6], ...; (step=3, tw=5) -> [0:5], [3:8], ...

    returns: sequences and targets for all sequences
    '''
    L = series_stft.shape[0]
    tw, pw = training_window, prediction_window
    stft_window = stft_params['noverlap']
    series = series[stft_window:]
    sequences, targets = [], []
    for i in range(0, L-tw-pw, step):
        # Get current sequence 
        sequences.append(series_stft[i:i+tw])
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
    

def create_dataloader(series, sequence_params, dataloader_params, shuffle=False):
    '''
    series: [np.ndarray] - time-series
    sequence_params: [dict] - {'training_window': 200, 'prediction_window': 100, 'step': 1}
    dataloader_params: [dict] - {'batch_size': 64, 'shuffle': True, 'drop_last': False}
    '''
    sequences, targets = generate_sequences(series, **sequence_params)
    dataset = SequenceDataset(sequences, targets)
    loader = DataLoader(dataset, shuffle=shuffle, **dataloader_params)
    return loader

def create_dataloader_stft_time(series, series_stft, sequence_params, dataloader_params,
                                stft_params, shuffle=False):
    '''
    series: [np.ndarray] - time-series
    series_stft: [np.ndarray] - time-series of STFT features 
    sequence_params: [dict] - {'training_window': 200, 'prediction_window': 100, 'step': 1}
    dataloader_params: [dict] - {'batch_size': 64, 'shuffle': True, 'drop_last': False}
    stft_params: [dict] - parameters for STFT (see scipy documentation)
    '''
    sequences, targets = generate_sequences_stft_time(series, series_stft, **sequence_params,
                                                      stft_params=stft_params)
    dataset = SequenceDataset(sequences, targets)
    loader = DataLoader(dataset, shuffle=shuffle, **dataloader_params)
    return loader
    
