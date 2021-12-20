import numpy as np
import os.path

import pandas as pd

import labels
import filePaths as fp
import houseTypes
from general import meta
import postProcessing as postProc

_xyName = meta.x_y(x="X", y="Y")
_mlDataName = meta.ml_data(train="Train", test="Test", validate="Validation")
_npSuffix = ".npy"

def write_og_like_df(df:pd.DataFrame, name):
    df.to_csv(fp.ogFormat + name.replace(' ', '') + ".csv", index=False)

def write_og_like_ml_data(data:meta.ml_data, name):
    write_og_like_df(data.train, name + _mlDataName.train)
    write_og_like_df(data.test, name + _mlDataName.test)

def write_og_like_files(realHome:houseTypes.house, fakeHome:houseTypes.house):
    fakeHome.maxTimeDif = realHome.maxTimeDif
    fakeUnnormed = postProc.back_to_real(fakeHome)
    realUnnormed = postProc.back_to_real(realHome)
    write_og_like_ml_data(fakeHome.data, fakeHome.name)
    write_og_like_ml_data(realUnnormed.data, realUnnormed.name)
    return fakeUnnormed, realUnnormed

#todo: np.savez

def _name_to_filename(name):
    return fp.numpyArrs + name.replace(' ', '') + _npSuffix

def _save_np_arr(arr:np.ndarray, name):
    np.save(_name_to_filename(name), arr)
    return name

def save_np_xy(xy:meta.x_y, name):
    _save_np_arr(xy.x, name + _xyName.x)
    _save_np_arr(xy.y, name + _xyName.y)

def save_np_ml_data(data:meta.ml_data, name):
    if data.train is not None: save_np_xy(data.train, name + _mlDataName.train)
    if data.test is not None: save_np_xy(data.test, name + _mlDataName.test)
    if data.validate is not None: save_np_xy(data.validate, name + _mlDataName.validate)

def save_np_house(home:houseTypes.house):
    save_np_ml_data(home.data, home.name)

def _load_np_arr(name):
    name = _name_to_filename(name)
    arr = None
    if os.path.exists(name):
        arr = np.load(name)
    return arr

def load_np_xy(name):
    xy = meta.x_y(
        x=_load_np_arr(name + _xyName.x),
        y=_load_np_arr(name + _xyName.y)
    )
    return xy

def load_ml_data(name):
    data = meta.ml_data(
        train=load_np_xy(name + _mlDataName.train),
        test=load_np_xy(name + _mlDataName.test),
        validate=load_np_xy(name + _mlDataName.validate)
    )
    return data

def load_house(name):
    return houseTypes.house(data=load_ml_data(name), name=name)