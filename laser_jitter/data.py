'''
This script provides data classes for time series analysis.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


__all__ = ['TimeSeries', 'TimeSeriesSTFT']


class TimeSeries:
    def __init__(self, ts, scaling='standard', smooth_params=None, train_size=0.8,
                 ):
        '''
        This class provides basic tools for operation with a single time series
        '''
        self.ts = ts
        self.smooth_params = smooth_params
        assert scaling in ['standard', 'minmax']
        self.scaling = scaling
        self.train_size = train_size

        if self.scaling is not None:
            self.scale(ts)

    def smooth(self, ts):
        ts_smooth = np.convolve(ts, self.smooth_params['kernel'], self.smooth_params['mode'])
        return ts_smooth

    def train_test_split(self, ts):
        train, test = train_test_split(ts, train_size=self.train_size, shuffle=False)
        return train, test

    def scale(self, ts, scaler=None):
        if scaler is None:
            scaler = StandardScaler() if self.scaling == 'standard' else MinMaxScaler((-1,1))
        if ts.ndim == 1:
            ts = ts[:,np.newaxis]
        ts_scaled = scaler.fit_transform(ts)
        return ts_scaled, scaler

    def split_and_scale(self, ts):
        train, test = self.train_test_split(ts)
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
            self.ts_smooth = self.smooth(self.ts)
            data_smooth = self.split_and_scale(self.ts_smooth)
            self.train_smooth, self.test_smooth, self.scaler_smooth = data_smooth.values()
            results['smoothed'] = data_smooth

            # Cut out smooth boundary region from unsmoothed timeseries
            # (useful for metric computation and comparison between smoothed and non-smoothed cases)
            N = self.smooth_params['N']
            self.ts = self.ts[N//2:len(self.ts)-N//2]

        data = self.split_and_scale(self.ts)
        self.train, self.test, self.scaler = data.values()
        results['unchanged'] = data
        return results
    
