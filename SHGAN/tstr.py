import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
import numpy as np
from sklearn import metrics as skm
import matplotlib.pyplot as plt
import math
import seaborn as sns

import genApi
import discApi
from general import meta, custActivations
import globalVars as gv
import filePaths as fp
import labels
import houseTypes
import postProcessing as postProc

_optimizer = 'adam'
_lossFunc = tf.keras.losses.MeanSquaredError()
_kernelInit = tf.keras.initializers.RandomNormal(stddev=0.02)
_batchSize = gv.BATCH_SIZE
_metrics = [tf.keras.metrics.CategoricalAccuracy()]

_epochs = 2 if meta.DEBUG else 100
_STEPS_PER_EPOCH = 2 if meta.DEBUG else None
_verbose = 1 #if meta.DEBUG else 0
_conditionalPivot = labels.start_stop(labels.pivots.activities.start, len(labels.conditionals))
#
# _epochs = 1
# _STEPS_PER_EPOCH = 2
# _verbose = 1

tstrModelFile = fp.kerasModel + "TimeCWTstr.km"


def tstr_model(featureSize=None):
    if featureSize is None:
        # featureSize = len(labels.colOrder) - len(labels.allActivities) #todo: bring back times
        featureSize = len(labels.colOrdinalDict)- len(labels.allActivities) - len(labels.conditionals)
    defArgs = {"use_bn":True, "use_bias":False, "use_dropout":False}
    inLayer = l.Input(shape=(gv.WINDOW_SIZE, featureSize))
    featureSizes = [featureSize]
    for _ in range(int(math.log2(gv.WINDOW_SIZE))-1):
        featureSizes.append(featureSizes[-1]*1.7)
    x = discApi.conv_block(inLayer, featureSizes[1], l.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT), **defArgs)
    for featSize in featureSizes[2:]:
        x = discApi.conv_block(x, featSize, l.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT), **defArgs)

    x = discApi.conv_block(x, len(labels.allActivities), custActivations.softmax_Neg1_1.activation)
    x = l.Flatten()(x)
    model = keras.models.Model(inLayer, x, name="TSTR_CGAN")
    return model

def tstr(data:meta.ml_data, model:keras.Model = None, savePath=None):
    if model is None: model =tstr_model(data.test.x.shape[-1])
    model.compile(_optimizer, tf.keras.losses.CategoricalCrossentropy(), metrics=_metrics)
    return fit_eval(model, data, savePath=savePath)


def fit_eval(model, data:meta.ml_data, epochs=_epochs, batchSize=_batchSize, verbose=_verbose,
             stepsPerEpoch = _STEPS_PER_EPOCH, savePath=None):
    cb = []
    if data.validate is not None:
        cb.append(keras.callbacks.EarlyStopping(monitor='val_loss', patience=3))

    # fit network
    history = model.fit(data.train.x, data.train.y, epochs=epochs, batch_size=batchSize, verbose=verbose,
                        steps_per_epoch=stepsPerEpoch, callbacks=cb,
                        # validation_data=(data.validate.x, data.validate.y) if data.validate is not None else None)
                        validation_data=(data.test.x, data.test.y))
    if meta.DEBUG:
        model.save(tstrModelFile)
    # evaluate model
    m = model.evaluate(data.test.x, data.test.y, batch_size=_batchSize, verbose=_verbose, return_dict=True)
    if savePath:
        model.save(savePath)
    preds = model(data.test.x)
    oneHotPreds = postProc.sensor_activity_one_hot(preds, falseValue=-1)
    acc = skm.accuracy_score(data.test.y, oneHotPreds)
    print(history.history)
    print("Accuracy:", acc)
    fig = plt.figure()
    ax = sns.lineplot(history.history['loss'])
    ax.set_title(savePath)
    return model, m, history, acc

def transform_xy_to_classifier_data(xy:meta.x_y)->meta.x_y:
    xyClass = meta.x_y(
        x=xy.x[...,:labels.pivots.activities.start],
        y=xy.x[:,-1,labels.pivots.activities.start:]
    )
    return xyClass

def transform_to_classifier_data(dataIn:meta.ml_data) -> meta.ml_data:
    data= meta.ml_data(
        train= transform_xy_to_classifier_data(dataIn.train),
        test= transform_xy_to_classifier_data(dataIn.test)
    )
    return data


def model_dispatcher(trainData:meta.x_y, testData:meta.x_y):
    tstrData = meta.ml_data(
        train=trainData,
        test=testData
    )
    tstrData = transform_to_classifier_data(tstrData)
    model = tstr_model()
    metrics = tstr(tstrData, model)
    return metrics

def trtr(data:meta.ml_data):
    model = tstr_model(data.train.x.shape[-1])
    model, metrics, history, acc, mae = tstr(data, model, savePath = None)
    print(metrics)
    print("One-hot acc:", acc)
    print("Original time MAE:", mae, "Normalized:", mae/2)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title("TRTR Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
