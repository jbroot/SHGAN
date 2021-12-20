import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as l
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest
import copy

import labels as lbl
import filePaths as fp
from Sum21.general import meta
import nnProcessing as nnpp
import genApi
import postProcessing as postProc
import houseTypes

_defPlotKwArgs = {"tight_layout":True}

_fileOutBuffer = ''

def print_file_out(fileName=None, clear=True):
    global _fileOutBuffer
    if fileName is None:
        fileName = fp.misc + "dataAnalysisDefaultFile.txt"
    with open(fileName, 'w+') as file:
        file.write(_fileOutBuffer)
    if clear:
        _fileOutBuffer = ''



def time_split_violin(realThenSynthetic, title=''):
    fig = plt.figure(tight_layout=True)
    ax = sns.violinplot(data=realThenSynthetic, split=True, cut=0).set_title(title)


def prob_y_given_x(collapsedDf, xs=lbl.allActivities, xName=lbl.rawLabels.activity, ys=lbl.allSensors,
                   yName=lbl.rawLabels.sensor):
    cpDf = pd.DataFrame(columns=xs)
    cpDf[yName] = ys
    cpDf.set_index(yName, inplace=True)
    for x in xs:
        yGivenX = collapsedDf.loc[collapsedDf[xName] == x, yName]
        if yGivenX.empty: continue
        cpDf[x] = yGivenX.value_counts(normalize=True)
    cpDf.fillna(0., inplace=True)
    return cpDf

def view_prob_x_given_y(xGivenY, name, xLabel, rotateX=0):
    fig = plt.figure(tight_layout=True, figsize=(10,8))
    plt.xticks(rotation=rotateX, ma='right')
    ax = sns.heatmap(xGivenY)
    ax.set_title(name)
    ax.set_xlabel(xLabel)
    if not meta.DEBUG:
        plt.savefig(fp.heatMapConditionals + name.replace(' ', '') + ".png", format='png')
    return ax

def view_signal_given_sensor(collapsedDf, name):
    forSignals = copy.deepcopy(collapsedDf)
    numBins = 20
    forSignals[lbl.rl.signal] = (forSignals[lbl.rl.signal] * numBins).round(decimals=0)/numBins
    signalNames = forSignals[lbl.rl.signal].unique()
    signalNames.sort()
    # signalNames = [str(x) for x in signalNames]
    sigGivenSens = prob_y_given_x(forSignals, xs=lbl.allSensors, xName=lbl.rl.sensor, ys=signalNames,
                                  yName=lbl.rl.signal)
    ax = view_prob_x_given_y(sigGivenSens, name + " Signal Given Sensor", lbl.rl.sensor)

    return ax

def view_interdependency(collapsedDf, name):
    sensGivenAct = prob_y_given_x(collapsedDf)
    view_prob_x_given_y(sensGivenAct, name + " Sensor Given Activity", lbl.rl.activity, rotateX=90)

    actGivenSens = prob_y_given_x(collapsedDf, xs=lbl.allSensors, xName=lbl.rl.sensor,
                                  ys=lbl.allActivities, yName=lbl.rl.activity)
    view_prob_x_given_y(actGivenSens, name + " Activity Given Sensor", lbl.rl.sensor)

    # view_signal_given_sensor(collapsedDf, name)
    return sensGivenAct, actGivenSens


def kolm_smirnov_analysis(data1, data2, nameSuffix):
    length = 10
    size = (length,length)
    plotArgs = {"tight_layout":True, "figsize":size}

    kssWindow, pvalsWindow = kolm_smirnov_by_window(data1, data2)
    kssFeatures = kssWindow.mean(axis=1)
    pvalsFeatures = pvalsWindow.mean(axis=1)
    fig = plt.figure(**plotArgs)
    cmap = sns.cubehelix_palette(as_cmap=True)
    ax = sns.kdeplot(kssFeatures, pvalsFeatures, cmap=cmap, fill=True)
    ax.set_title("KS Scores and Respective P-Values for " + nameSuffix)
    ax.set_xlabel("Kolmogorov-Smirnov Scores")
    ax.set_ylabel("Two-Sided P-Values")
    if not meta.DEBUG:
        plt.savefig(fp.ksTests + nameSuffix.replace(' ','') + "Kde.png")

    fig = plt.figure(tight_layout=True, figsize=(10,7))
    ax = sns.boxplot(kssFeatures)
    ax.set_title("Kolmogorov-Smirnov Scores for " + nameSuffix)
    if not meta.DEBUG:
        plt.savefig(fp.ksTests + nameSuffix.replace(' ','') + "Boxplot.png")

    def round(val):
        return str(np.around(val, decimals=3))

    if meta.DEBUG:
        print(nameSuffix, "KS Mean", round(kssFeatures.mean()), "STD", round(kssFeatures.std()))
        print(nameSuffix, "P-vals Mean", round(pvalsFeatures.mean()), "STD", round(pvalsFeatures.std()))
    colDelim = " & "
    global _fileOutBuffer
    _fileOutBuffer += nameSuffix + colDelim + round(kssFeatures.mean()) + colDelim + round(kssFeatures.std()) \
                      + colDelim + round(pvalsFeatures.mean()) + colDelim + round(pvalsFeatures.std()) + '\\\\\n'
    return kssWindow, pvalsWindow

def kolm_smirnov_by_feature(data1, data2):
    assert data1.shape == data2.shape
    nFeatures = data1.shape[-1]
    kss, pvals = np.zeros(shape=(nFeatures)), np.zeros(shape=(nFeatures))
    for i in range(nFeatures):
        kss[i], pvals[i] = kstest(data1[..., i].flatten(), data2[..., i].flatten())
    return kss, pvals

def kolm_smirnov_by_window(data1, data2):
    assert data1.shape == data2.shape
    nSamples = data1.shape[0]
    nFeatures = data1.shape[-1]
    kss = np.zeros(shape=(nSamples, nFeatures))
    pvals = np.zeros(shape=(nSamples, nFeatures))
    for i in range(nSamples):
        kss[i], pvals[i] = kolm_smirnov_by_feature(data1[i], data2[i])
        # if meta.DEBUG:
        #     break
    return kss, pvals


def quantitative_analyses(realData:meta.x_y, genOut:meta.x_y, name):
    kmStats = kolm_smirnov_analysis(realData.x, genOut.x, name)
    return kmStats

def compare_houses_quantitative(firstN):
    allHomes = nnpp.get_windows_by_house(firstN=firstN)
    assert len(allHomes) == 3
    xs = []
    for home in allHomes:
        xs.append(np.concatenate((home.data.train.x, home.data.test.x)))

    ksStats = []
    for h1Pos, h2Pos in ((0,1),(1,2),(0,2)):
        name = "Homes " + allHomes[h1Pos].name + " and " + allHomes[h2Pos].name
        smallerInstances = min(xs[h1Pos].shape[0], xs[h2Pos].shape[0])
        ksStats.append(
            kolm_smirnov_analysis(data1=xs[h1Pos][:smallerInstances], data2=xs[h2Pos][:smallerInstances], nameSuffix=name)
        )

    return ksStats


def compare_train_test(home:houseTypes.house, firstN=None):
    minInstances =  min(home.data.train.x.shape[0], home.data.test.x.shape[0])
    firstN = minInstances if firstN is None else min(firstN, minInstances)
    return kolm_smirnov_analysis(data1=home.data.train.x[:firstN], data2=home.data.test.x[:firstN],
                                 nameSuffix="Different Days of Home " + home.name)

def compare_houses_with_selves(firstN=None):
    allHomes = nnpp.get_windows_by_house(firstN=firstN)
    rets = []
    for home in allHomes:
        rets.append(compare_train_test(home))
    return rets

def contrast_rnd_uniform_noise(data:np.ndarray, name):
    dMin = data.min()
    dMax = data.max()
    noise = np.random.uniform(dMin, dMax, size=data.shape)
    noise = postProc.sensor_activity_one_hot(noise, falseValue=dMin)
    noise[...,lbl.pivots.signal.start] = np.random.choice([dMin,dMax], size=noise.shape[:-1])
    noise = postProc.enforce_alt_signal_each_sensor(noise)
    return kolm_smirnov_analysis(data, noise, name)

def compare_real_synthetic(realHome:houseTypes.house, fakeHome:houseTypes.house, name):
    quantStats = meta.ml_data(
        train= quantitative_analyses(realHome.data.train, fakeHome.data.train, name + " Training Data"),
        test = quantitative_analyses(realHome.data.test, fakeHome.data.test, name + " Testing Data")
    )
    return quantStats

def count_portions(collapsedDf, doActivities:bool):
    def get_portion(name, index):
        return collapsedDf[name].value_counts(normalize=True).reindex(index, fill_value=0)

    sensors = get_portion(lbl.rl.sensor, lbl.allSensors)
    if doActivities:
        activities = get_portion(lbl.rl.activity, lbl.allActivities)
        portions = pd.concat((sensors,activities))
        return portions
    return sensors

def view_portions(df1, name1, df2, name2, doActivities=True):
    genPortions = count_portions(df1, doActivities)
    realPortions = count_portions(df2, doActivities)
    fig, axes = plt.subplots(2, 1, figsize=(15,10),  **_defPlotKwArgs)
    axes = axes.flatten()
    for ax, portion, name in zip(axes, (genPortions, realPortions), (name1, name2)):
        sns.barplot(x=portion.index, y=portion.values, ax=ax).set_title(name + " proportions")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    if meta.DEBUG:
        plt.savefig(fp.barPlots + name1 + name2 + "barplots.png")
    return genPortions, realPortions


def get_history_rename():
    histRename = {
        'loss': "Loss", 'mean_absolute_error': "Mean Absolute Error",
        'cosine_similarity': "Cosine Similarity", 'categorical_crossentropy': "Categorical Crossentropy"
    }
    histRenameAux = tuple(histRename.items())
    for key, value in histRenameAux:
        histRename["val_" + key] = "Validation " + value
    return histRename


def plot_history(history, name):
    histRename = get_history_rename()
    metrics = { histRename[key] : np.reshape(np.asarray(value), (-1,1)) for key,value in history.history.items()}
    metricTrainScores = [key for key in metrics.keys() if "Validation " not in key]
    metricValScores = [key for key in metrics.keys() if "Validation " in key]
    dfs = [pd.DataFrame(np.concatenate((metrics[train], metrics[val]), axis=-1), columns=[train, val])
           for train, val in zip(metricTrainScores, metricValScores)]

    for df, trainKey in zip(dfs, metricTrainScores):
        assert trainKey in df.columns
        fig = plt.figure()
        df.reset_index(inplace=True, drop=True)
        ax = sns.lineplot(data = df)
        title = trainKey + " for " + name
        ax.set_title(title)
        ax.set_ylabel(trainKey)
        ax.set_xlabel("Epochs")
        if not meta.DEBUG:
            plt.savefig(fp.tstr + title.replace(' ', ''))

    return metrics

if __name__ == "__main__":
    pass