import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import globalVars as gv
import cnnApi

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
    x = layers.Conv1D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x

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


def get_cDisc():
    return get_discriminator_model()


def get_discriminator_model(shapeIn = gv.IMG_SHAPE):
    discDropOut = gv.DROPOUT_DEFAULT
    img_input = layers.Input(shape=shapeIn)
    x = layers.GaussianNoise(stddev=.1)(img_input)
    # fSizes = cnnApi.scale_cells_linearly(shapeIn[-1], )
    x = conv_block(
        x,
        110,
        strides=2,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
        use_dropout=True,
        drop_value=discDropOut,
    )

    x = conv_block(
        x,
        200,
        strides=2,
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
        use_dropout=True,
        drop_value=discDropOut,
    )
    x = conv_block(
        x,
        400,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
        use_bias=True,
        use_dropout=True,
        drop_value=discDropOut,
    )

    x = conv_block(
        x,
        800,
        strides=2,
        use_bn=False,
        activation=layers.LeakyReLU(gv.LEAKY_ALPHA_DEFAULT),
        use_bias=True,
        use_dropout=True,
        drop_value=discDropOut,
    )

    x = layers.Flatten()(x)
    x = layers.Dropout(discDropOut)(x)
    x = layers.Dense(1)(x)

    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model
