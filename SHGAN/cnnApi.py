import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def scale_cells_linearly(initFeatureSize, targetFeatureSize, timeSteps, includeFirst=False):
    sizes = np.linspace(
        start=initFeatureSize * timeSteps[0], stop=targetFeatureSize * timeSteps[-1], num=len(timeSteps)
    )
    sizes = (sizes / timeSteps).round()
    if includeFirst:
        return sizes
    return sizes[1:]

