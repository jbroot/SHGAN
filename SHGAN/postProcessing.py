import math
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance as wDist
from sklearn.metrics.pairwise import cosine_similarity as cosSimF

import labels as lbl
import globalVars as gv
import houseTypes
import genApi
from Sum21.general import meta
import nnProcessing as nnpp

def get_time_delta(series:pd.Series, val=0):
    return pd.Timedelta(val, units=series.dtype.char)

_time_epsilon = pd.Timedelta(1, 'ms')

def collapse_sensor_and_activity(df:pd.DataFrame, removeOld=True):
    df[lbl.rl.sensor] = df[list(lbl.allSensors)].idxmax(axis=1)
    df[lbl.rl.activity] = df[list(lbl.allActivities)].idxmax(axis=1)
    if removeOld:
        df.drop(list(lbl.allSensors + lbl.allActivities), axis=1, inplace=True)
    return df


def to_df(windows, columns, saveFile=''):
    idx = np.arange(0, windows.shape[1])
    idx = np.tile(idx, windows.shape[0])
    windows = np.reshape(windows, (-1, windows.shape[-1]))
    df = pd.DataFrame(windows, columns=columns, index=idx)
    if saveFile:
        df.to_csv(saveFile)
    # df[lbl.rl.time] = pd.to_timedelta(df[lbl.rl.time])
    return df

def sensor_activity_one_hot(windows, falseValue=0):
    #for each row, get the max in features and in labels
    windows = sensor_to_one_hot(windows, falseValue)
    windows = activity_to_one_hot(windows, falseValue)
    return windows

def one_hot_row(row, colRange, falseValue=0):
    subRow = row[colRange[0]:colRange[1]]
    argmax = np.argmax(subRow)
    subRow = np.zeros(subRow.shape)
    if falseValue != 0:
        subRow.fill(falseValue)
    subRow[argmax] = 1
    if colRange == (None,None):
        return subRow
    row[colRange[0]:colRange[1]] = subRow
    return row

def to_one_hot(windows, colRange=(None,None), falseValue=0):
    return np.apply_along_axis(one_hot_row, axis=-1, arr=windows,
                               colRange=colRange, falseValue=falseValue)

def sensor_to_one_hot(windows, falseValue=0):
    return to_one_hot(windows, colRange=(lbl.pivots.sensors.start, lbl.pivots.sensors.stop), falseValue=falseValue)

def activity_to_one_hot(windows, falseValue=0):
    return to_one_hot(windows, colRange=(lbl.pivots.activities.start, lbl.pivots.activities.stop), falseValue=falseValue)

def enforce_alt_signal_each_sensor(arr3d:np.ndarray, rng=gv.MIN_MAX_RNG):
    #sensors must be one-hot encoded
    for feat in lbl.allSensors:
        featOrdinal = lbl.colOrdinalDict[feat]
        for i in range(arr3d.shape[0]):
            mask = arr3d[i, :, featOrdinal] == 1
            if mask.any():
                # if mask.sum() > 1:
                #     breakpoint()
                arr3d[i, mask, lbl.colOrdinalDict[lbl.rl.signal]]\
                    = enforce_alt_binary(arr3d[i, mask, lbl.colOrdinalDict[lbl.rl.signal]], rng)
    # falseMask = arr3d[...,lbl.colOrdinalDict[lbl.rl.signal]] == rng[0]
    # if rng[0] != 0:
    #     arr3d[falseMask, lbl.colOrdinalDict[lbl.rl.signal]] = 0
    # if rng[1] != 1:
    #     arr3d[~falseMask, lbl.colOrdinalDict[lbl.rl.signal]] = 1
    return arr3d

def enforce_alt_binary(arr, rng):
    option1 = np.tile((rng[0], rng[1]), math.ceil(arr.shape[0]/2))
    option2 = np.tile((rng[1], rng[0]), math.ceil(arr.shape[0]/2))
    if arr.shape[0] % 2:
        option1, option2 = option1[:-1], option2[:-1]

    cosSim1 = cosSimF([arr], [option1])
    # cosSim2 = cosSimF([arr], [option2])
    assert cosSim1.shape == (1,1)
    return option1 if cosSim1 > 0 else option2

def fix_gen_out(arr, rng=gv.MIN_MAX_RNG):
    arr = sensor_activity_one_hot(arr, falseValue=rng[0])
    arr = enforce_alt_signal_each_sensor(arr, rng)
    return arr

def get_all_fixed_synthetic(realData:houseTypes.house, cgan, rng=gv.MIN_MAX_RNG, name="Fake Home"):
    fakeHome = houseTypes.house(
        data=meta.ml_data(
            train=meta.x_y(),
            test=meta.x_y()
        ),
        name = name,
        maxTimeDif=realData.maxTimeDif
    )
    def get_fixed_data(y):
        xy = genApi.get_cgen_xy(cgan.generator, y)
        xy.x = fix_gen_out(xy.x, rng)
        return xy

    fakeHome.data.train = get_fixed_data(realData.data.train.y)
    fakeHome.data.test = get_fixed_data(realData.data.test.y)
    return fakeHome

def binarize(df:pd.DataFrame, minMax = None):
    if minMax is None:
        minMax = (df.min(), df.max())
    greaterMask = df > (minMax[1] + minMax[0])/2
    df[greaterMask] = 1
    df[~greaterMask] = 0
    return df

def collapse_raw_x(arr3d:np.ndarray)->pd.DataFrame:
    arr3d = sensor_activity_one_hot(arr3d)
    arr3d = enforce_alt_signal_each_sensor(arr3d) #todo: fix not alternating
    df = to_df(arr3d, lbl.features)
    collapsedDf = collapse_sensor_and_activity(df)
    collapsedDf = collapsedDf[lbl.rl.correctOrder]
    return collapsedDf


def unnorm_time(xy:meta.x_y, oldTimeMax):
    time = xy.x[...,lbl.colOrdinalDict[lbl.rl.time]]
    time = nnpp.unnorm_time(time, oldTimeMax)
    time = time.cumsum(axis=-1)

    addToEachWindow =nnpp.inverse_norm_per_day(xy.y[...,lbl.colOrdConditional[lbl.timeMidn]])
    day = np.argmax(xy.y[...,1:], axis=-1)
    day *= pd.Timedelta(1, 'd').value
    addToEachWindow += day
    firstDate = pd.to_datetime(["2010-01-04 00:00:00"]).values.astype(addToEachWindow.dtype)
    addToEachWindow += firstDate
    addToEachWindow = addToEachWindow[:,np.newaxis]
    addToEachWindow = np.repeat(addToEachWindow, xy.x.shape[1], axis=-1)
    xy.x[...,lbl.colOrdinalDict[lbl.rl.time]] = time + addToEachWindow.astype(time.dtype)
    return xy

def unnorm_time_mldata(data:meta.ml_data, oldTimeMax):
    if data.train is not None: data.train = unnorm_time(data.train, oldTimeMax)
    if data.test is not None: data.test = unnorm_time(data.test, oldTimeMax)
    if data.validate is not None: data.validate = unnorm_time(data.validate, oldTimeMax)
    return data

def unnorm_time_house(home:houseTypes.house):
    home.data = unnorm_time_mldata(home.data, home.maxTimeDif)
    return home

def _back_to_real_xy(xy:meta.x_y):
    df = collapse_raw_x(xy.x)
    df[lbl.rl.time] = pd.to_datetime(df[lbl.rl.time])
    df.loc[df[lbl.rl.signal] == -1, [lbl.rl.signal]] = 0

    df.loc[df[lbl.rl.sensor].isin(lbl.doorSensors), [lbl.rl.signal]] = \
        df.loc[df[lbl.rl.sensor].isin(lbl.doorSensors), [lbl.rl.signal]]\
            .replace({0:lbl.doorFalse,1:lbl.doorTrue})

    df.loc[df[lbl.rl.sensor].isin(lbl.motionSensors), [lbl.rl.signal]] = \
        df.loc[df[lbl.rl.sensor].isin(lbl.motionSensors), [lbl.rl.signal]]\
            .replace({0:lbl.motionFalse,1:lbl.motionTrue})

    return df


def back_to_real(home:houseTypes.house):
    home = unnorm_time_house(home)
    home.data.train = _back_to_real_xy(home.data.train)
    home.data.test = _back_to_real_xy(home.data.test)
    return home


def is_time_ordered(time: pd.Series):
    diff = time.diff()
    diff = diff < pd.Timedelta(0, unit=diff.dtype.char)
    return diff.sum() <= 1 #first time is always false

def order_synthetic_time(fakeHome:houseTypes.house):
    print("Order time")
    fakeHome = back_to_real(fakeHome)
    def order_df(df:pd.DataFrame):
        df.reset_index(inplace=True)

        #equal vals with time epsilon
        sanityCheck = 0
        while True:
            assert sanityCheck < gv.WINDOW_SIZE
            #get where time is less than epsilon, except for start of windows
            mask = (df[lbl.rl.time].diff() < _time_epsilon) & (df['index'] != 0)
            print("Checking time epsilon. Iter:", sanityCheck, "\nMask size:", mask.sum(), '\n')
            if not mask.any():
                break
            df.loc[mask, lbl.rl.time] += _time_epsilon
            sanityCheck += 1

        def get_decreasing():
            diff = df[lbl.rl.time].diff()
            decreasing = diff < get_time_delta(diff)
            print("Decreasing window steps. iter:", iter, "\nDecreasing size:", decreasing.sum(), '\n')
            decreasing = decreasing[decreasing].index
            decreasing = np.repeat(decreasing, gv.WINDOW_SIZE)
            decreasing = np.reshape(decreasing, (-1, gv.WINDOW_SIZE))
            first = np.arange(gv.WINDOW_SIZE)[None, :]
            second = np.zeros(decreasing.shape[0])[:, None].astype(decreasing.dtype)
            decreasing += (first + second)
            return decreasing.flatten()
        iter=0
        while len(decreasing := get_decreasing()) > 0:
            df.drop(decreasing, inplace=True)
            iter += 1

        df.index = df['index']
        df.drop(['index'], axis=1, inplace=True)
        # assert is_time_ordered(df[lbl.rl.time])
        return df

    fakeHome.data.transform(order_df)
    return fakeHome
