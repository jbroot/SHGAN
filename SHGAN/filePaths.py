from Sum21.general import meta
import globalVars as gv

dataPath = "datasets/"
rawPath = dataPath + "raw/"
processedPath = dataPath + "processed/"
misc = "misc/"
experimental = misc + "experimental/"
if gv.EXPERIMENTAL: misc = experimental
tstr = misc + "tstr/"
kerasModel = misc + "kerasModels/"
qualitativeAnalyses = misc + "qualitativeAnalyses/"
ksTests = qualitativeAnalyses + "ksTests/"
heatMapConditionals = qualitativeAnalyses + "heatMapConditionals/"
barPlots = qualitativeAnalyses + "barplots/"
numpyArrs = misc + "numpyArrs/"
losses = misc + "losses/"
dataframes = misc + "dataframes/"
ogFormat = misc + "originalFormat/"

casasBinHome = "casasBinaryHome/"

def get_all_casas_bin_paths(path = processedPath):
    if processedPath in path:
        fileReTrain = "%sb%dTrain.csv"
        fileReTest = "%sb%dTest.csv"
    elif rawPath in path:
        fileReTrain = "%sb%d.train.csv"
        fileReTest = "%sb%d.test.csv"
    else:
        raise ValueError("Path not recognized.")

    homePaths = [
        meta.ml_data(train, test) for train, test in (
            (fileReTrain % (path, i), fileReTest % (path, i)) for i in range(1, 4)
        )
    ]
    return homePaths

