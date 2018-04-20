from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.TrainModelExperiment import TrainModelExperiment
from experiment.SaveModelTrainingExperiment import SaveModelTrainingExperiment

import dataset.loaderPackaged as loaders
import config.packagedConfigs as modelConfigs


datasetModelEpochsBatchSizeTuples = [
    (loaders.shapesTransformedPackage, modelConfigs.conv_64x64_7_16_256_32_bce, 0, 1000)
]

for loaderPackage, modelConfig, epochs, batchSize in datasetModelEpochsBatchSizeTuples:
    buildModel = BuildModelExperiment(modelConfig)
    builtModel = buildModel.run()
    
    trainModel = TrainModelExperiment(builtModel, loaderPackage.datasetLoader, epochs, batchSize)
    modelTrainingHistory = trainModel.run()

    saveModelTrainingHistory = SaveModelTrainingExperiment(builtModel, modelTrainingHistory, loaderPackage.name + "_" + modelConfig.stringDescriptor)
    saveModelTrainingHistory.run()
