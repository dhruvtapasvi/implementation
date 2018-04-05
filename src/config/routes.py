ROOT = ".."

RESOURCE_ROUTE = ROOT + "/res"

MODEL_WEIGHTS_ROUTE = ROOT + "/cacheWeights"
MODEL_WEIGHTS_EXTENSION = ".weights.h5"
def getModelWeightsRoute(name):
    return MODEL_WEIGHTS_ROUTE + "/" + name + MODEL_WEIGHTS_EXTENSION

MODEL_TRAINING_HISTORY_ROUTE = ROOT + "/modelTrainingHistory"
MODEL_TRAINING_HISTORY_EXTENSION = ".history.p"
def getModelTrainingHistoryRoute(name):
    return MODEL_TRAINING_HISTORY_ROUTE + "/" + name + MODEL_TRAINING_HISTORY_EXTENSION

OUT_ROUTE = ROOT + "/out"
def getResultRouteStem(name):
    return OUT_ROUTE + "/" + name + "_"

CONFIG_ROUTE = ROOT + "/config"
def getConfigRoute(name):
    return CONFIG_ROUTE + "/" + name
