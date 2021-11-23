import tensorflow as tf
import keras
import numpy as np

def get_bias_major_weights(model):
    weights = model.get_weights()
    biasMajor = []
    for arrI in range(0, len(weights), 2):
        inWeights = weights[arrI]
        biasWeights = weights[arrI+1].reshape((1,-2))
        l = np.concatenate((biasWeights, inWeights), axis=0).T
        biasMajor.append(l)
    return np.asarray(biasMajor)


def get_max_arg_vals(arr3D):
    amaxes = tf.argmax(arr3D, axis=-1)
    windowIdx = np.arange(0, amaxes.shape[0])
    rowIdx = np.arange(0, amaxes.shape[1])
    return arr3D[windowIdx[:, np.newaxis], rowIdx[np.newaxis, :], amaxes]

def get_steps_per_epoch(nSamplesOg, fracOfOg):
    return int(max(nSamplesOg * fracOfOg), 1)

def get_steps_and_epochs(nSamplesOg, fracOfOg, epochsIfFull):
    stepsPerEpoch = get_steps_per_epoch(nSamplesOg, fracOfOg)
    epochs = int(max(epochsIfFull / fracOfOg, 1))
    return stepsPerEpoch, epochs
