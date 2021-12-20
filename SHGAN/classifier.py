import copy

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as l
from sklearn import metrics as skm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Sum21.general import meta
from Sum21.general import custActivations as custActs
import globalVars as gv
import nnProcessing as nnpp
import postProcessing as postProc
import houseTypes
import filePaths
import dataAnalysis as da



DATA_AMT = int(1e5) if meta.DEBUG else None
NTIMESTEPS = 32
_DROPOUT_DEFAULT = 0.25
_LEAKY_ALPHA_DEFAULT = 0.2
_OPTIMIZER = keras.optimizers.Adam(2e-4, 0.5)
# _LOSSFUNC = keras.losses.MeanSquaredError()
_LOSSFUNC = keras.losses.CategoricalCrossentropy()
# _KERNELINIT = keras.initializers.RandomNormal(stddev=0.02)
_METRICS = [keras.metrics.MeanAbsoluteError(),
            keras.losses.CosineSimilarity(),
            keras.losses.CategoricalCrossentropy()]

def conv_block(
    x,
    filters,
    activation,
    kernel_size=2,
    strides=2,
    padding="same",
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=gv.DROPOUT_DEFAULT,
):
    x = l.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = l.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = l.Dropout(drop_value)(x)
    return x

def get_cnn(nFeatures, nLabels): #todo: fix hardcode filters
    inputLayer = keras.Input(shape=(NTIMESTEPS, nFeatures))
    x = conv_block(inputLayer, filters=40, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = conv_block(x, filters=32, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = conv_block(x, filters=24, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = conv_block(x, filters=19, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = conv_block(x, filters=nLabels, activation=keras.activations.softmax, use_bn=False)
    x = l.Flatten()(x)
    model = keras.models.Model(inputLayer, x)
    model.compile(loss = _LOSSFUNC, optimizer=_OPTIMIZER, metrics=_METRICS)
    return model

def analyze_model(test:meta.x_y, model, name, history, plot=True):
    print("Analyzing", name)
    pred = model(test.x)
    oneHot = postProc.to_one_hot(pred, falseValue=0)
    acc = skm.accuracy_score(allHomesConcat.data.test.y, oneHot)
    print("Accuracy:", acc)
    if plot:
        metrics = da.plot_history(history, name)
        return metrics


def train_time_series(allHomesConcat, model, epochs, name,):
    print("Training", name)
    stepsPerEpoch = 2 if meta.DEBUG else None
    # callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=1)]
    callbacks = []
    # assert (allHomesConcat.data.train.x != allHomesConcat.data.test.x).all()
    history = model.fit(
        allHomesConcat.data.train.x,
        allHomesConcat.data.train.y,
        validation_data=(allHomesConcat.data.test.x, allHomesConcat.data.test.y),
        steps_per_epoch=stepsPerEpoch,
        epochs=epochs,
        callbacks=callbacks
    )

    if not meta.DEBUG and name:
        model.save(filePaths.kerasModel + name + ".km")
    return model, history

def rnn_block(
    x,
    units,
    activation,
    rnnLayer = l.GRU,
    use_bias=True,
    use_bn=False,
    use_dropout=False,
    drop_value=gv.DROPOUT_DEFAULT,
    use_max_pool=True,
    use_avg_pool=False,
    pool_size=2,
    strides=2,
    padding='same',
):
    x = rnnLayer(
        units=units, use_bias=use_bias, return_sequences=True
    )(x)
    if use_bn:
        x = l.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = l.Dropout(drop_value)(x)
    if use_max_pool:
        x = l.MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding)(x)
    if use_avg_pool:
        x = l.AvgPool1D(pool_size=pool_size, strides=strides, padding=padding)(x)
    return x

def get_rnn(nFeatures, nLabels): #todo: fix hardcode filters
    inputLayer = keras.Input(shape=(NTIMESTEPS, nFeatures))
    x = rnn_block(inputLayer, units=40, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = rnn_block(x, units=32, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = rnn_block(x, units=24, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = rnn_block(x, units=19, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = rnn_block(x, units=nLabels, activation=keras.activations.softmax, use_bn=False)
    x = l.Flatten()(x)
    model = keras.models.Model(inputLayer, x)
    model.compile(loss = _LOSSFUNC, optimizer=_OPTIMIZER, metrics=_METRICS)
    return model

def get_bi_rnn(nFeatures, nLabels): #todo: fix hardcode filters
    inputLayer = keras.Input(shape=(NTIMESTEPS, nFeatures))
    backwardsArgs = {"return_sequences":True, "go_backwards":True, "use_bias":True}
    x = l.GRU(units=40, **backwardsArgs)(inputLayer)
    x = rnn_block(x, units=40, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = l.GRU(units=24, **backwardsArgs)(x)
    x = rnn_block(x, units=24, activation= l.LeakyReLU(_LEAKY_ALPHA_DEFAULT), use_bn=True)
    x = l.GRU(units=nLabels, **backwardsArgs)(x)
    x = rnn_block(x, units=nLabels, activation=keras.activations.softmax, use_bn=False)
    x = l.Flatten()(x)
    model = keras.models.Model(inputLayer, x)
    model.compile(loss = _LOSSFUNC, optimizer=_OPTIMIZER, metrics=_METRICS)
    return model

def get_mlp(nFeatures, nLabels):
    inputLayer = keras.Input(shape=NTIMESTEPS*nFeatures)
    x = l.Dense(32 * NTIMESTEPS)(inputLayer)
    x = l.LeakyReLU(_LEAKY_ALPHA_DEFAULT)(x)
    x = l.BatchNormalization()(x)
    x = l.Dense(21 * NTIMESTEPS)(x)
    x = l.LeakyReLU(_LEAKY_ALPHA_DEFAULT)(x)
    x = l.BatchNormalization()(x)

    x = l.Dense(nLabels)(x)
    x = keras.activations.softmax(x)

    model = keras.models.Model(inputLayer, x)
    model.compile(loss = _LOSSFUNC, optimizer=_OPTIMIZER, metrics=_METRICS)
    return model

def train_mlp(allHomesConcat, model, epochs, name):
    mlpData = houseTypes.house(allHomesConcat.data, name)
    mlpData.data.transform(nnpp.flatten_x)
    return train_time_series(mlpData, model, epochs, name)


#todo: visualize time features with a violin plot

#todo later: add noise over features before predicting

def get_nn_filename(name):
    return filePaths.tstr + name +".km"

def do_models(houseData:houseTypes.house, nameSuffix):

    nFeatures, nLabels = houseData.data.train.x.shape[-1], houseData.data.train.y.shape[-1]
    cnnModel = get_cnn(nFeatures, nLabels)
    # cnnModel, cnnHistory = train_time_series(houseData, cnnModel, epochs = 5 if meta.DEBUG else 5, name= "CNN " + nameSuffix)
    cnnModel, cnnHistory = train_time_series(houseData, cnnModel, epochs = 3, name= "CNN " + nameSuffix)
    cnnMetrics = analyze_model(houseData.data.test, cnnModel, "CNN " + nameSuffix, cnnHistory)

    rnnModel = get_rnn(nFeatures, nLabels,)
    # rnnModel, rnnHistory = train_time_series(houseData, rnnModel, epochs = 2 if meta.DEBUG else 10, name= "RNN " + nameSuffix)
    rnnModel, rnnHistory = train_time_series(houseData, rnnModel, epochs = 3, name= "RNN " + nameSuffix)
    rnnMetrics = analyze_model(houseData.data.test, rnnModel, "RNN " + nameSuffix, rnnHistory)
    # biRnn = get_bi_rnn(nFeatures, nLabels,)
    # birnn = train_time_series(allHomesConcat, biRnn, epochs = 2 if meta.DEBUG else 20, name= "Bi RNN")
    mlp = get_mlp(nFeatures, nLabels)
    mlpHomes = copy.deepcopy(houseData)
    # mlp, mlpHistory = train_mlp(mlpHomes, mlp, epochs= 2 if meta.DEBUG else 5, name= "MLP " + nameSuffix)
    mlp, mlpHistory = train_mlp(mlpHomes, mlp, epochs= 3, name= "MLP " + nameSuffix)
    mlpMetrics = analyze_model(mlpHomes.data.test, mlp, "MLP " + nameSuffix, mlpHistory)

    fig = plt.figure()
    df = {}

    for network, metric in zip(("CNN ", "RNN ", "MLP "), (cnnMetrics, rnnMetrics, mlpMetrics)):
        for prefix in ("Validation ", ""):
            name = prefix + network
            key = prefix + "Categorical Crossentropy"
            df[name + "Categorical Crossentropy"] = metric[key].flatten()
    df = pd.DataFrame(df)
    df.reset_index(inplace=True)
    plt.figure()
    sns.lineplot(data = df[df['index']==0].melt(id_vars='index'), x='index', y='value', hue='variable')

    if not meta.DEBUG:
        nameOut = nameSuffix.replace(' ', '')
        metricPrint = "cnn" + nameOut + "Metrics = " + str(cnnMetrics) + "\nrnn" + nameOut + "Metrics = " + str(
            rnnMetrics) + \
                      "\nmlp" + nameOut + "Metrics = " + str(mlpMetrics)

        # metricPrint = "cnn" + nameOut + "Metrics = " + str(cnnMetrics) + \
        #               "\nmlp" + nameOut + "Metrics = " + str(mlpMetrics)

        with open(filePaths.misc + "historyDicts" + nameSuffix.replace(' ','') + ".txt", "w+") as f:
            f.write(metricPrint)

if __name__ == "__main__":
    allHomesConcat = nnpp.get_windows_activity_label(firstN=500 if meta.DEBUG else None)

    do_models(allHomesConcat, "All Data")

    # removedFeatureHomes = copy.deepcopy(allHomesConcat)
    # removedFeatureHomes.data.train.x = removedFeatureHomes.data.train.x[...,1:]
    # removedFeatureHomes.data.test.x = removedFeatureHomes.data.test.x[...,1:]
    # do_models(removedFeatureHomes, "Removed Time Differentials")
    #
    # removedFeatureHomes = copy.deepcopy(allHomesConcat)
    #
    # removedFeatureHomes.data.train.x = np.concatenate(
    #     (removedFeatureHomes.data.train.x[...,0, np.newaxis], removedFeatureHomes.data.train.x[...,2:]), axis=-1)
    # removedFeatureHomes.data.test.x = np.concatenate(
    #     (removedFeatureHomes.data.test.x[...,0,np.newaxis], removedFeatureHomes.data.test.x[...,2:]), axis=-1)
    # do_models(removedFeatureHomes, "Removed Sensor Signals")
    #
    # removedFeatureHomes = copy.deepcopy(allHomesConcat)
    # removedFeatureHomes.data.train.x = removedFeatureHomes.data.train.x[...,2:]
    # removedFeatureHomes.data.test.x = removedFeatureHomes.data.test.x[..., 2:]
    # do_models(removedFeatureHomes, "Removed Sensors")

    plt.show()
    exit()