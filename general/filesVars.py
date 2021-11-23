
class dir:
    misc = "misc/"
    datasets = "datasets/"
    kerasModels = misc + "kerasModels/"
    autoencoders = kerasModels + 'autoencoder/'

    @staticmethod
    def get_autoencoder_dir_of_latent_size(latentSize):
        return dir.autoencoders + "ae" + str(latentSize) + '/'

class dataset:
    _directory = dir.datasets
    _processedDir = _directory + "processed/"
    _rawDir = _directory + "raw/"
    _normDir = _directory + "normalized/"
    _npDir = _directory + "numpy/"
    synsysSubDir = "synsys/"
    synsysFileName = "synsysData.csv" #1 time dim; 84 unique sensors; 25 activities

    @staticmethod
    def _fix_dir_str(dir):
        return dir + '/' if dir[-1] != '/' else dir

    @staticmethod
    def get_raw(fileName, subDir = synsysSubDir):
        subDir = dataset._fix_dir_str(subDir)
        return dataset._rawDir + subDir + fileName
    @staticmethod
    def get_processed(fileName, subDir = synsysSubDir):
        subDir = dataset._fix_dir_str(subDir)
        return dataset._processedDir + subDir + fileName
    @staticmethod
    def get_normalized(fileName, subDir = synsysSubDir):
        subDir = dataset._fix_dir_str(subDir)
        return dataset._normDir + subDir + fileName

    @staticmethod
    def get_np(fileName, subDir = synsysSubDir):
        subDir = dataset._fix_dir_str(subDir)
        return dataset._npDir + subDir + fileName