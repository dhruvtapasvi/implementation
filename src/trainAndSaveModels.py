from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.TrainModelExperiment import TrainModelExperiment
from experiment.SaveModelTrainingExperiment import SaveModelTrainingExperiment

import dataset.loaderPackaged as loaders
import config.packagedConfigs as modelConfigs
from experiment.ExperimentalConfigTuple import ExperimentalConfigTuple
from experiment.experimentalConfigTuples import experimentalConfigTuples as prepackagedExperimentalTuples


experimentalTuples = [
    ExperimentalConfigTuple(loaders.mnistTransformedLimitedRotationPackage, modelConfigs.deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce, 500, 100),
    ExperimentalConfigTuple(loaders.mnistTransformedLimitedRotationPackage, modelConfigs.conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce, 500, 100),
    ExperimentalConfigTuple(loaders.shapesTransformedLimitedRotationPackage, modelConfigs.deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce, 500, 100),
    ExperimentalConfigTuple(loaders.shapesTransformedLimitedRotationPackage, modelConfigs.conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce, 500, 100)
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
