from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.TrainModelExperiment import TrainModelExperiment
from experiment.SaveModelTrainingExperiment import SaveModelTrainingExperiment

import dataset.loaderPackaged as loaders
import config.packagedConfigs as modelConfigs
from experiment.ExperimentalConfigTuple import ExperimentalConfigTuple
from experiment.experimentalConfigTuples import experimentalConfigTuples as prepackagedExperimentalTuples


experimentalTuples = [
    ExperimentalConfigTuple(loaders.shapesTransformedPackage, modelConfigs.conv_64x64_7_16_256_32_bce, 1000, 0)
]

for experimentalTuple in experimentalTuples:
    buildModel = BuildModelExperiment(experimentalTuple.config)
    builtModel = buildModel.run()

    try:
        trainModel = TrainModelExperiment(builtModel, experimentalTuple.datasetPackage.datasetLoader, experimentalTuple.epochs, experimentalTuple.batchSize)
        modelTrainingHistory = trainModel.run()
    except TypeError:
        print("NoneType error observed in training:", experimentalTuple.stringDescriptor)
    except:
        print("An unexpected error observed in training:", experimentalTuple.stringDescriptor)
        continue

    try:
        saveModelTrainingHistory = SaveModelTrainingExperiment(builtModel, modelTrainingHistory, experimentalTuple.stringDescriptor)
        saveModelTrainingHistory.run()
    except:
        print("Error occured while saving weights for:", experimentalTuple.stringDescriptor)
