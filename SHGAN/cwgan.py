import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as l
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import labels

from general import meta
import nnProcessing as nnpp
import globalVars as gv
import filePaths as fp
import discApi
import genApi
import dataAnalysis as da
import tstr
import postProcessing as postProc
import labels as lbl
import houseTypes
import postProcIO as ppIO

meta.tf_np_behavior()

cgeneratorFile = fp.kerasModel + "TimeCWGen.km"
cdiscFile = fp.kerasModel + "TimeCWDisc.km"

LOAD_FILES = True

GAN_EPOCHS= 2 if meta.DEBUG else 10 #will be 20
STEPS_PER_EPOCH=2 if meta.DEBUG else None
DATA_AMT = int(1e3) if meta.DEBUG else None
# DATA_AMT = None
DATA_AMT_PER_HOME = int(1e2) if meta.DEBUG else None

#
# GAN_EPOCHS= 2
# STEPS_PER_EPOCH=2
# DATA_AMT = int(1e5)



def get_tf_data(npData:meta.ml_data):
    allData = meta.ml_data()
    allData.train = tf.data.Dataset.from_tensor_slices((npData.train.x, npData.train.y))
    allData.train = allData.train.shuffle(buffer_size=1024).batch(gv.BATCH_SIZE)
    allData.test = tf.data.Dataset.from_tensor_slices((npData.test.x, npData.test.y))
    allData.test = allData.test.shuffle(buffer_size=1024).batch(gv.BATCH_SIZE)
    return allData

class CWGAN(keras.Model):
    discLossKey = "dLoss"
    genLossKey = "gLoss"

    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(CWGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(CWGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, dataset): #todo: keep same latent space for each epoch
        features, conditionals = dataset
        data = meta.x_y(features, conditionals)
        ogY = data.y
        data.y = data.y[:,None,:]
        data.y = tf.repeat(data.y, repeats=tf.shape(data.x)[1], axis=1)
        xyConcat = tf.concat((data.x, data.y), axis=-1)
        # Get the batch size
        batch_size = tf.shape(data.x)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator
        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                genOut = genApi.get_cgen_out(
                    self.generator, ogY, float64=True, training=True)
                genOut = tf.concat((genOut, data.y), axis=-1)
                # Get the logits for the fake images
                fake_logits = self.discriminator(genOut, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(xyConcat, training=True)
                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, xyConcat, genOut)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight
            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        # Train the generator
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = genApi.get_cgen_out(self.generator, data.y[:,0,:],
                                                         float64=True, training=True)
            generated_images = tf.concat((generated_images, data.y), axis=-1)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
            gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
            self.g_optimizer.apply_gradients(
                zip(gen_gradient, self.generator.trainable_variables)
            )
        return {"dLoss": d_loss, "gLoss": g_loss}

    def fit(self, *args, **kwargs):
        self.history = super(CWGAN, self).fit(*args, **kwargs)

    def plot_losses(self, name="cganLossDefaultName.png"):
        plt.plot(self.history.history[CWGAN.discLossKey])
        plt.plot(self.history.history[CWGAN.genLossKey])
        plt.hlines(0, 0, len(self.history.history[CWGAN.discLossKey]), 'k', 'dashed')
        plt.title("CWGAN Losses")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend([CWGAN.discLossKey, CWGAN.genLossKey])
        if not meta.DEBUG:
            plt.savefig(fp.losses + name)

    def call(self, input):
        #only for keras' validation
        return self.discriminator(input)


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    # return -tf.reduce_mean(fake_img)
    mean = -tf.reduce_mean(fake_img)
    # homoLoss = cLoss.rm_homo_bin_sens_pairs(genOut)
    return mean
    # return -cLoss.arctan_mse_like(fake_img)


def get_cgan(cgen, disc, trainData, fit=True):

    cgan = CWGAN(
        discriminator=disc, generator=cgen, latent_dim=gv.NOISE_DIM,
        discriminator_extra_steps=1 if meta.DEBUG else 5
    )
    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )

    cgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )
    if fit:
        cgan.fit(trainData, batch_size=gv.BATCH_SIZE, epochs=GAN_EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
        if not meta.DEBUG:
            cgan.generator.save(cgeneratorFile)
            cgan.discriminator.save(cdiscFile)

    return cgan

def train_vary_disc_epochs():
    allHomesConcat, cgan, fakeHome = get_data_cgan(fit=False)
    trainData = allHomesConcat.data.train.x
    if isinstance(trainData, np.ndarray):
        trainData = get_tf_data(allHomesConcat.data)
    nEpochs=1
    for i in range(0, GAN_EPOCHS, nEpochs):
        print("Epoch", i, "of", GAN_EPOCHS, end=': ')
        cgan.fit(trainData.train, batch_size=gv.BATCH_SIZE, epochs=nEpochs, steps_per_epoch=STEPS_PER_EPOCH)
        if abs(cgan.history.history[CWGAN.genLossKey][-1]) > 10:
            cgan.d_steps = max(1, cgan.d_steps - 1)
        if abs(cgan.history.history[CWGAN.discLossKey][-1]) > 7:
            cgan.d_steps += 1

        if not meta.DEBUG and i % 50 == 0:
            cgan.generator.save(cgeneratorFile)
            cgan.discriminator.save(cdiscFile)

    if not meta.DEBUG:
        cgan.generator.save(cgeneratorFile)
        cgan.discriminator.save(cdiscFile)
    return allHomesConcat, cgan

def load_cgen_cdisc():
    cgen = keras.models.load_model(cgeneratorFile)
    cdisc = keras.models.load_model(cdiscFile)
    return cgen, cdisc

def qual_test_helper(data1, data1Name, data2, data2Name):
    print("Interdependency heatmaps")
    genCollapsed = postProc.collapse_raw_x(data2)
    realCollapsed = postProc.collapse_raw_x(data1)
    # times = pd.concat((realCollapsed[labels.rl.time], genCollapsed[labels.rl.time]))
    # da.time_split_violin(times, "Normalized Times")
    da.view_interdependency(genCollapsed, data1Name)
    da.view_interdependency(realCollapsed, data2Name)
    return realCollapsed, genCollapsed

def qualitative_tests(data1, data1Name, data2, data2Name):
    df1, df2 = qual_test_helper(data1, data1Name, data2, data2Name)
    print("Bar plots")
    da.view_portions(df1, data1Name, df2, data2Name)
    return df1, df2

def get_data_cgan(firstN=DATA_AMT, fit=False):
    allHomesConcat = nnpp.get_all_concat_windows(firstN=firstN)
    print("train.x shape:", allHomesConcat.data.train.x.shape)
    print("train.y shape:", allHomesConcat.data.train.y.shape)

    allDataTf = get_tf_data(allHomesConcat.data) if fit else None
    if LOAD_FILES:
        generator, discriminator = load_cgen_cdisc()
    else:
        discriminator = discApi.get_cDisc()
        generator = genApi.get_cgen()

    cgan = get_cgan(generator, discriminator, None if allDataTf is None else allDataTf.train, fit=fit)
    if fit: cgan.plot_losses()
    return allHomesConcat, cgan

# def write_og_like_files(realHome:houseTypes.house, fakeHome:houseTypes.house):
#     fakeHome.maxTimeDif = realHome.maxTimeDif
#     fakeUnnormed = postProc.back_to_real(fakeHome)
#     realUnnormed = postProc.back_to_real(realHome)
#     fakeUnnormed.data.train.to_csv(fp.ogFormat + fakeHome.name.replace(' ', '') + ".csv", index=False)
#     realUnnormed.data.train.to_csv(fp.ogFormat + realHome.name.replace(' ', '') + "Train.csv", index=False)
#     realUnnormed.data.test.to_csv(fp.ogFormat + realHome.name.replace(' ', '') + "Test.csv", index=False)
#     return fakeUnnormed, realUnnormed

def run_ks_tests():
    print("Running KS tests")
    # print("Different Houses")
    # da.compare_houses_quantitative(DATA_AMT_PER_HOME)
    # print("Houses on themselves (different days)")
    # da.compare_houses_with_selves(DATA_AMT_PER_HOME)
    print("Getting CGAN")
    allHomesConcat, cgan, fakeHome = get_data_cgan(DATA_AMT, fit=False)
    print("Real and Synthetic Data")
    da.compare_real_synthetic(allHomesConcat, fakeHome, "Synthetic versus Real")
    # print("Noise versus real data")
    # da.contrast_rnd_uniform_noise(allHomesConcat.data.train.x, "Real Data and Uniform Noise")
    print("Noise versus synthetic data")
    da.contrast_rnd_uniform_noise(fakeHome.data.train.x, "Synthetic Data and Uniform Noise")
    if not meta.DEBUG:
        da.print_file_out(fp.ksTests + "meanStdTable.txt")
    return allHomesConcat, fakeHome, cgan

def run_tstr():
    allHomesConcat, cgan, fakeHome = get_data_cgan(DATA_AMT, fit=False)
    tstrData = meta.ml_data(
        train=fakeHome.data.train,
        test=allHomesConcat.data.train
    )
    tstrData = tstr.transform_to_classifier_data(tstrData)
    model = tstr.tstr_model()
    metrics = tstr.tstr(tstrData, model, "tstr")

    trtrData = tstr.transform_to_classifier_data(allHomesConcat.data)
    modelReal = tstr.tstr_model()
    metricsReal = tstr.tstr(trtrData, modelReal, "trtr")
    return

def do_analyses():
    allHomesConcat, fakeHome, cgan = run_ks_tests()
    # allHomesConcat, cgan, fakeHome = get_data_cgan(DATA_AMT, fit=False)
    qualitative_tests(allHomesConcat.data.train.x, "Real", fakeHome.data.train.x, "Synthetic")
    # run_tstr()
    return allHomesConcat, fakeHome, cgan


if __name__ == "__main__":
    # if meta.DEBUG: meta.enable_tf_debug(eager=True, debugMode=True)
    # else: meta.enable_tf_debug(eager=False, debugMode=False)
    meta.enable_tf_debug(eager=False, debugMode=False)

    # allHomesConcat, cgan = train_vary_disc_epochs()
    # do_analyses()
    allHomesConcat, cgan= get_data_cgan(DATA_AMT, fit=False)
    fakeHome = postProc.get_all_fixed_synthetic(allHomesConcat, cgan)
    fakeHome = postProc.order_synthetic_time(fakeHome)
    if not meta.DEBUG:
        ppIO.write_og_like_ml_data(fakeHome.data, fakeHome.name)

    # plt.show()

    exit()