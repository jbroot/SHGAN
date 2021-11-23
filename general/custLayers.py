import keras
from keras import layers
import tensorflow as tf
import numpy as np

class mono_inc_rnn_cell(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(mono_inc_rnn_cell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units


    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel',)# constraint=tf.keras.constraints.NonNeg())
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel', constraint=keras.constraints.NonNeg())
        # self.trans_out_kernel = self.add_weight(
        #     shape=(self.units, self.units),
        #     initializer='uniform',
        #     name='recurrent_kernel', constraint=tf.keras.constraints.NonNeg())
        # self.transition_weight = self.add_weight(
        #     shape = (self.units,1), initializer='uniform', constraint=tf.keras.constraints.NonNeg()
        # )
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.matmul(inputs, self.kernel)
        output = h + tf.matmul(prev_output, self.recurrent_kernel)
        return output, [output]


class Argmax(layers.Layer):
    """
    Based on https://github.com/YerevaNN/R-NET-in-Keras/blob/master/layers/Argmax.py
    """
    def __init__(self, axis=-1, **kwargs):
        super(Argmax, self).__init__(**kwargs)
        # self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return keras.backend.argmax(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        del input_shape[self.axis]
        return tuple(input_shape)

    # def compute_mask(self, x, mask):
    #     return None

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Argmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == "__main__":
    # Let's use this cell in a RNN layer:

    cell = mono_inc_rnn_cell(32)
    x = keras.Input((None, 5))
    layer = layers.RNN(cell, return_sequences=True)
    y = layer(x)

    # Here's how to use the cell to build a stacked RNN:

    cells = [mono_inc_rnn_cell(32), mono_inc_rnn_cell(64)]
    x = keras.Input((None, 5))
    layer = layers.RNN(cells)
    y = layer(x)
    exit(0)
