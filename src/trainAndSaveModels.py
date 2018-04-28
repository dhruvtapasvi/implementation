from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.TrainModelExperiment import TrainModelExperiment
from experiment.SaveModelTrainingExperiment import SaveModelTrainingExperiment

from experiment.experimentalConfigTuples import experimentalConfigTuples as prepackagedExperimentalTuples


experimentalTuples = prepackagedExperimentalTuples

for experimentalTuple in experimentalTuples:
    buildModel = BuildModelExperiment(experimentalTuple.config)
    builtModel = buildModel.run()

    try:
        trainModel = TrainModelExperiment(builtModel, experimentalTuple.datasetPackage.datasetLoader, experimentalTuple.epochs, experimentalTuple.batchSize)
        modelTrainingHistory = trainModel.run()
    except TypeError as te:
        print("NoneType error observed in training:", experimentalTuple.stringDescriptor)
        print(te)
    except Exception as e:
        print("An unexpected error observed in training:", experimentalTuple.stringDescriptor)
        print(e)
        continue

    try:
        saveModelTrainingHistory = SaveModelTrainingExperiment(builtModel, modelTrainingHistory, experimentalTuple.stringDescriptor)
        saveModelTrainingHistory.run()
    except Exception as e:
        print("Error occured while saving weights for:", experimentalTuple.stringDescriptor)
        print(e)
