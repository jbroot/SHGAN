import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import globalVars as gv
from general import meta, custActivations as custAct
import labels
# import discApi
# import cnnApi
import math

blockDefArgs = {"up_size": 2, "strides": 1, "use_bias": False, "use_bn": True, "padding": "same", "use_dropout": False}

def get_gen_input(shape=(gv.BATCH_SIZE, gv.NOISE_DIM), minMax = gv.MIN_MAX_RNG):
    #todo: clip
    data = tf.random.uniform(shape=shape, minval=minMax[0], maxval=minMax[1])
    return data

def get_cgen_out(cgenerator, labels, noiseDim = gv.NOISE_DIM, float64=True, training=False):
    dtype = "float64" if float64 else "float32"
    noise = get_gen_input((tf.shape(labels)[0], noiseDim)).astype(dtype)
    return cgenerator([labels.astype(dtype), noise], training=training).astype(dtype)

def get_cgen_xy(cGenerator, conditionals):
    return meta.x_y(get_cgen_out(cGenerator, conditionals).numpy(), conditionals)


def get_cgen_concatted(cgen, labels):
    assert len(labels.shape) == 2
    cgenOut = get_cgen_out(cgen, labels)
    y = labels[:, np.newaxis, :]
    y = tf.repeat(y, repeats=cgenOut.shape[1], axis=1)
    return tf.concat((cgenOut, y), axis=-1)


# def get_cgen_concatted(cgenerator, labels, **kwargs):
#     cgenOut = get_cgen_out(cgenerator, labels, **kwargs)
#     return tf.concat([cgenOut, labels], axis=-1)

def bi_lstm(x, units, use_dropout=False, use_bn=False, drop_value=0.3):
    forward = layers.LSTM(units, return_sequences=True)
    backward = layers.LSTM(units, go_backwards=True, return_sequences=True)
    x = layers.Bidirectional(forward, backward_layer=backward)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(0.2)(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x

def upsample_block(
    x,
    filters,
    activation,
    kernel_size=2,
    strides=1,
    up_size=2,
    padding="same",
    use_bn=False,
    use_bias=True,
    use_dropout=False,
    drop_value=gv.DROPOUT_DEFAULT,
):
    if use_bn:
        use_bias=False

    if up_size > 1: x = layers.UpSampling1D(up_size)(x)
    x = layers.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x

def get_cgen(nLabelsIn=gv.LABEL_SIZE, noiseDim=gv.NOISE_DIM):
    #target size is N X windowsize X feature size = None X 32 X 48
    targetInputShape = (1, 256)
    targetInputShapeSize = targetInputShape[0] * targetInputShape[1]
    upScaleInSize = (math.floor(targetInputShapeSize/2), math.ceil(targetInputShapeSize/2))
    labelsIn = layers.Input(shape=nLabelsIn) #todo scale conditionals and noise space
    noiseIn = layers.Input(shape=noiseDim)
    lx = layers.Dense(upScaleInSize[0], use_bias=True)(labelsIn)
    nx = layers.Dense(upScaleInSize[1], use_bias=True)(noiseIn)
    x = layers.Concatenate()([lx, nx])
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT)(x)
    x = layers.Reshape(targetInputShape)(x)
    x = upsample_block(
        x,
        192,
        layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
        up_size=2,
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        128,
        layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
        up_size=2,
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        96,
        layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
        up_size=2,
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        66,
        layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
        up_size=2,
        strides=1,
        use_bias=False,
        use_bn=True,
        padding="same",
        use_dropout=False,
    )
    realVals = upsample_block(
        x,
        labels.pivots.sensors.start,
        keras.activations.tanh,
        up_size=2,
        strides=1,
        use_bias=True,
        use_bn=False,
        padding="same",
        use_dropout=False,)
    oneHotSensors = upsample_block(
        x,
        len(labels.allSensors),
        custAct.softmax_Neg1_1.activation,
        up_size=2,
        strides=1,
        use_bias=True,
        use_bn=False,
        padding="same",
        use_dropout=False,
    )
    oneHotActivities = upsample_block(
        x,
        len(labels.allActivities),
        custAct.softmax_Neg1_1.activation,
        up_size=2,
        strides=1,
        use_bias=True,
        use_bn=False,
        padding="same",
        use_dropout=False,
    )
    x = layers.Concatenate()([realVals, oneHotSensors, oneHotActivities])
    model = keras.models.Model([labelsIn, noiseIn], x, name="ConditionalGenerator")
    return model

#for 2d labels
# def get_cgen(nTimeSteps = trainTest.TRAIN_SHAPE.nTimeSteps, nLabelsIn=trainTest.TRAIN_SHAPE.nGanFeatures-2, noiseDim=gv.COND_NOISE_DIM,
#              sigmoidSpace=False):
#     targetGenInShape = (1,32)
#     labelTargetDim=targetGenInShape[1]-noiseDim
#
#     featuresEachEncStep = np.linspace(nLabelsIn, labelTargetDim, 7)
#
#     #encode labels
#     encIn = layers.Input(shape=(nTimeSteps, nLabelsIn))
#     encX = discApi.conv_block(encIn, featuresEachEncStep[1], strides=2, activation=layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT))
#     for nFeatures in featuresEachEncStep[2:-1]:
#         encX = discApi.conv_block(encX, nFeatures, strides=2, activation=layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT))
#
#     rvAct = keras.activations.sigmoid if sigmoidSpace else keras.activations.tanh
#     encX = discApi.conv_block(encX, labelTargetDim, strides=2, activation=rvAct)
#     # encX = layers.Flatten()(encX)
#
#     genIn = layers.Input(shape=(1,noiseDim,))
#     x = layers.Concatenate()((encX, genIn))
#     # x = layers.Reshape(targetGenInShape)(x)
#     x = upsample_block(
#         x,
#         24,
#         layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
#         up_size=2,
#         strides=1,
#         use_bias=False,
#         use_bn=True,
#         padding="same",
#         use_dropout=False,
#     )
#     x = upsample_block(
#         x,
#         16,
#         layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
#         up_size=2,
#         strides=1,
#         use_bias=False,
#         use_bn=True,
#         padding="same",
#         use_dropout=False,
#     )
#     x = upsample_block(
#         x,
#         10,
#         layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
#         up_size=2,
#         strides=1,
#         use_bias=False,
#         use_bn=True,
#         padding="same",
#         use_dropout=False,
#     )
#     x = upsample_block(
#         x,
#         6,
#         layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
#         up_size=2,
#         strides=1,
#         use_bias=False,
#         use_bn=True,
#         padding="same",
#         use_dropout=False,
#     )
#     x = upsample_block(
#         x,
#         3,
#         layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
#         up_size=2,
#         strides=1,
#         use_bias=False,
#         use_bn=True,
#         padding="same",
#         use_dropout=False,
#     )
#     x = upsample_block(
#         x,
#         2,
#         keras.activations.tanh,
#         up_size=2,
#         strides=1,
#         use_bias=True,
#         use_bn=False,
#         padding="same",
#         use_dropout=False,
#     )
#     model = keras.models.Model([encIn, genIn], x, name="ConditionalGenerator")
#     return model
