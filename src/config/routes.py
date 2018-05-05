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

RESULTS_ROUTE = ROOT + "/results"
def getRecordedResultsRoute(name):
    return RESULTS_ROUTE + "/" + name


USER_STUDY_ROUTE = ROOT + "/userStudy"
def getUserStudyRoute(name):
    return USER_STUDY_ROUTE + "/" + name

USER_STUDY_RESPONSES_ROUTE = ROOT + "/userStudyResponses"
def getUserStudyResponsesRoute(name):
    return USER_STUDY_RESPONSES_ROUTE + "/" + name
