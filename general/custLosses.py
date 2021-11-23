import tensorflow as tf
from synSys import sensorMetaData as sc, preprocessing as pp


def arctan_mse_like(x:tf.Tensor):
    return tf.reduce_mean(
        tf.add(
            tf.multiply(
                tf.atan(
                    tf.divide(
                        tf.pow(x, 3),
                    100)
                ),
            20),
        x)
    )

def _get_argmax(x:tf.Tensor):
    tf.argmax(x)

def rm_homo_bin_sens_pairs(x:tf.Tensor):
    #how add to loss?
    sensors = tf.add(tf.argmax(x[:,:,pp.synsysPivots.xOtherPos0:pp.synsysPivots.actPos0], axis=-1), pp.synsysPivots.xOtherPos0)
    mask = tf.greater_equal(sensors, pp.synsysPivots.xBinPos0)
    binX = tf.boolean_mask(x, mask)
    #todo:finish

