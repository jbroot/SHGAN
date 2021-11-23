import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from general import meta
import globalVars as gv
import nnProcessing as nnpp


DATA_AMT = int(1e5) if meta.DEBUG else None


def get_data(dataAmt=DATA_AMT):
    allDataNp, maxTimeDif = nnpp.get_windows(dataAmt)
    print("train.x shape:", allDataNp.train.x.shape)
    print("train.y shape:", allDataNp.train.y.shape)
    allData = meta.ml_data()
    allData.train = tf.data.Dataset.from_tensor_slices((allDataNp.train.x, allDataNp.train.y))
    allData.train = allData.train.shuffle(buffer_size=1024).batch(gv.BATCH_SIZE)
    allData.test = tf.data.Dataset.from_tensor_slices((allDataNp.test.x, allDataNp.test.y))
    allData.test = allData.test.shuffle(buffer_size=1024).batch(gv.BATCH_SIZE)
    return allDataNp, allData, maxTimeDif

#todo: data is {x,y} for x= (time differential, signal, sensor (32 one-hot), activity (14 one-hot))
# and y= (time from midnight, weekday (7 one-hot))
# For classification, needs to be {x,y} for x = time differential, signal, sensor (32 one-hot), time from midnight, weekday (7 one-hot)
# and y = activity (14 one-hot)
# graph metrics

#todo: collapse activity such that there's one activity per window, not one per time step.
# choose the final activity in the window to be that label

#todo: after previous todos, can create and train a classifier

#todo: visualize time features with a violin plot

#todo later: add noise over features before predicting

#other
#todo: maxTimeDif is a lonely attribute (messy) only used for un-normalizing the time differences

if __name__ == "__main__":
    allDataNp, allData, maxTimeDif = get_data()

    exit()