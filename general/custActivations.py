import numpy as np
import keras
from keras import layers
import tensorflow as tf
from abc import ABCMeta, abstractmethod

class optTanh:
    _coeff = 1.7159
    name = 'optTanh'
    @staticmethod
    def activation(x):
        return optTanh._coeff * np.tanh(x * 2 / 3)

    rng = (-_coeff, _coeff)


class thermoBetaSoftMax:
    name = 'thermoBetaSoftMax'
    @staticmethod
    def activation(x):
        return tf.nn.softmax(300 * x)

    rng = (0,1)

class thermoBetaSoftMaxNeg1_1:
    name = thermoBetaSoftMax.name + "Neg1_1"
    @staticmethod
    def activation(x):
        return 2*thermoBetaSoftMax.activation(x)-1

class softmax_Neg1_1:
    name = 'softmax' + "Neg1_1"
    @staticmethod
    def activation(x):
        return tf.subtract(
            tf.multiply(
                tf.nn.softmax(
                    # tf.multiply(x,2)
                    x
                ),
                2
            ),
            1
        )

        # return 2 * tf.nn.softmax(2*x) - 1

keras.utils.generic_utils.get_custom_objects().update(
    {thermoBetaSoftMax.name : layers.Activation(thermoBetaSoftMax.activation)}
)

def get_activation_layer(act_t:type):
    return keras.layers.Activation(act_t.activation, name=act_t.name)