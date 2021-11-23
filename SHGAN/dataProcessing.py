import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp


class min_max:
    def __init__(self, feature_range=(-1,1)):
        self.targetMin = feature_range[0]
        self.targetMax = feature_range[1]
        self.targetRange = self.targetMax - self.targetMin

    def fit_transform(self, arr):
        if isinstance(arr, pd.DataFrame):
            arr = arr.to_numpy()
        self.ogMin = np.min(arr)
        self.ogMax = np.max(arr)
        self.ogRange = self.ogMax - self.ogMin
        arr = (self.targetRange) * (arr - self.ogMin)/self.ogRange + self.targetMin
        return arr

    def inverse_transform(self, arr, onlyRealNums=False):
        # clipFactor = 0
        # minClip = self.ogMin - self.ogMin * clipFactor
        # maxClip = self.ogMax + self.ogMax * clipFactor
        # zeroIndex = np.where(arr == 0)
        arr = self.ogRange*(arr - self.targetMin)/(self.targetRange) + self.ogMin
        if any_inf_or_nan(arr):
            msg = "Nan or Infinite values found."
            if onlyRealNums:
                raise ValueError(msg)
            else:
                Warning(msg)
        # arr = arr.clip(minClip, maxClip)
        return arr

def any_inf_or_nan(arr):
    return not np.all(np.isfinite(arr)) or np.any(np.isnan(arr))

def min_max_vals(df, rng=(-1, 1)):
    scaler = min_max(feature_range=rng)
    if len(df.shape) == 1:
        arr = df.values.reshape(-1,1)
        arr = scaler.fit_transform(arr)
    else:
        arr = scaler.fit_transform(df)
    if isinstance(df, pd.DataFrame):
        df = pd.DataFrame(arr, index=df.index, columns=df.columns)
        return df, scaler
    elif isinstance(df, pd.Series):
        df = pd.Series(arr.reshape((-1,)), index=df.index, name=df.name)
        return df, scaler
    return arr, scaler

def min_max_each_col(df:pd.DataFrame, rng=(-1,1)):
    for col in df.columns:
        df[col], _ = min_max_vals(df, rng)
    return df