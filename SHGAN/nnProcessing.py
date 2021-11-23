import copy

import pandas as pd
import numpy as np
from general import meta

import preprocessing as pp
import dataProcessing as dp
import labels
import globalVars as gv
import houseTypes

def window_stack(arr, width=32, stepsize=None):
    if stepsize is None:
        stepsize = int(width/2)

    nWindows = int(arr.shape[0] / stepsize -1)
    indexer = np.arange(width)[None, :] + stepsize*np.arange(nWindows)[:,None]
    arr = arr[indexer]
    return arr


_timeExp = .3
# _timeExp = .09

def unnorm_time(arr, oldMax):
    arr = ((arr+1)/2)**(1/_timeExp) * oldMax
    assert (arr >= 0).all()
    return arr

def norm_time(arr):
    arr -= arr.min()
    arrMax = arr.max()
    arr /= arrMax
    arr = 2*arr**_timeExp - 1
    assert not (arr < -1).any()
    assert not (arr > 1).any()
    return arr, arrMax

class normed_df:
    def __init__(self, df, minMaxRng= (-1,1), doAsserts=True):
        self.df = df
        self.minMaxRng = minMaxRng
        self.doAsserts = doAsserts
        self.binMap = {0:minMaxRng[0], 1:minMaxRng[1]}
        self.threeDArr = None
        self.maxTimeDif = -1

    def norm_binaries(self):
        for col in labels.allBinaryColumns:
            self.df[col] = self.df[col].map(self.binMap)
        if self.doAsserts: assert self.df.loc[:, labels.allBinaryColumns].isin(self.minMaxRng).all().all()

    def days_week_time_midnight(self):
        tmp = pd.DataFrame(columns=labels.week + [labels.rl.time])
        self.df[labels.timeMidn] = self.df[labels.rl.time]
        tmp[labels.rl.time] = pd.to_datetime(self.df[labels.rl.time])
        w = tmp[labels.rl.time].dt.day_name()
        w = pd.get_dummies(w)
        tmp[w.columns] = w
        tmp.fillna(value=0, inplace=True)
        #to time from midnight
        self.df[labels.timeMidn] = (tmp[labels.rl.time] - tmp[labels.rl.time].dt.normalize()).astype(int)
        self.df[labels.rl.time] = self.df[labels.timeMidn]
        self.df = pd.concat((self.df, tmp[labels.week]), axis=1)
        return

    def time_difs_windows(self, windowSize=gv.WINDOW_SIZE, shuffle=False, norm=True):
        assert (self.df.columns == labels.correctOrder).all()
        self.df[labels.rl.time] = self.df[labels.rl.time].diff()
        self.df.reset_index(inplace=True, drop=True) #index on different houses
        dayStarts = list(self.df[self.df[labels.rl.time] < 0].index) + [self.df.shape[0]-1]
        dayStartsCpy = copy.deepcopy(dayStarts)
        dayStartsCpy.sort()
        assert dayStartsCpy == dayStarts
        if len(dayStarts) == 1:
            dayStarts = [0, dayStarts[0]]
        windows = None
        for i in range(len(dayStarts)-1):
            begin = dayStarts[i] + 1
            end = dayStarts[i+1]
            assert (self.df.iloc[begin+1:end][labels.rl.time] >= 0).all()
            windowsToConcat = window_stack(self.df.iloc[begin+1:end].to_numpy(), windowSize)
            if windows is None:
                windows = windowsToConcat
            else:
                windows = np.concatenate((windows, windowsToConcat), axis=0)
        assert (windows[..., labels.colOrdinalDict[labels.rl.time]] >= 0).all()
        if shuffle: np.random.shuffle(windows)

        xy = meta.x_y(x=windows[...,:len(labels.features)], y=windows[..., 0, len(labels.features):])
        if norm:
            xy.x[...,labels.colOrdinalDict[labels.rl.time]], self.maxTimeDif = norm_time(xy.x[...,labels.colOrdinalDict[labels.rl.time]])
            # assert (windows[..., labels.colOrdinalDict[labels.rl.time]] >= 0).all()
            normPerDay = 8.64e13
            assert not (windows > normPerDay).any()
            def norm_since_mid(arr):
                return ((arr / normPerDay) - .5 ) * 2
            xy.y[...,0] = norm_since_mid(xy.y[...,0])
        return xy

def inverse_norm_per_day(timeArr):
    timeArr = ((timeArr / 2) + 0.5) * 8.64e13
    return timeArr

def get_windows(data=None, name=''):
    data.transform(normed_df)
    data.apply(normed_df.days_week_time_midnight)
    data.apply(normed_df.norm_binaries)
    windows = data.apply(normed_df.time_difs_windows)
    return houseTypes.house(windows, name, data.train.maxTimeDif)

def get_all_concat_windows(firstN=None, name=labels.home_names.allHomes):
    return get_windows(pp.all_homes_concat(firstN), name=name)

def get_windows_by_house(firstN=None):
    allHomes = pp.get_all_homes(firstN=firstN)
    homeWindows = []
    for i, home in enumerate(allHomes):
        homeWindows.append(get_windows(home, name="H" + str(i+1)))
    return homeWindows


if __name__ == "__main__":
    windows, maxTimeDif = get_windows(firstN=int(1e5) if meta.DEBUG else None)
    exit(0)