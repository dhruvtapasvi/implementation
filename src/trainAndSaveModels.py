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
    
    trainModel = TrainModelExperiment(builtModel, experimentalTuple.datasetPackage.datasetLoader, experimentalTuple.epochs, experimentalTuple.batchSize)
    modelTrainingHistory = trainModel.run()

    saveModelTrainingHistory = SaveModelTrainingExperiment(builtModel, modelTrainingHistory, experimentalTuple.stringDescriptor)
    saveModelTrainingHistory.run()
