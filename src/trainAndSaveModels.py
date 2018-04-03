from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
from config.DenseAutoencoderConfig import DenseAutoencoderConfig

from dataset.loader.basic.MnistLoader import MnistLoader
from dataset.loader.basic.NorbLoader import NorbLoader
from dataset.loader.basic.MnistTransformedLoader import MnistTransformedLoader
from dataset.loader.preprocess.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne

from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.TrainModelExperiment import TrainModelExperiment
from experiment.SaveModelTrainingExperiment import SaveModelTrainingExperiment


resourcesRoot = "../res"
mnistLoader = ScaleBetweenZeroAndOne(MnistLoader(), 0, 255)
norbLoader = ScaleBetweenZeroAndOne(NorbLoader(resourcesRoot + "/norb"), 0, 255)
mnistTransformedLoader = ScaleBetweenZeroAndOne(MnistTransformedLoader(resourcesRoot + "/mnistTransformed_10"), 0, 255)

modelConfigRoot = "../config/model"
modelConfigs = [
    # (modelConfig, DatasetLoader, epochs, batchSize)
    (ConvolutionalAutoencoderConfig(modelConfigRoot + "/mnist_transformed_conv_7_16_256_32_bce.json"), mnistTransformedLoader, 100, 1000)
]

for modelConfig, datasetLoader, epochs, batchSize in modelConfigs:
    buildModel = BuildModelExperiment(modelConfig)
    builtModel = buildModel.run()

    trainModel = TrainModelExperiment(builtModel, datasetLoader, epochs, batchSize)
    modelTrainingHistory = trainModel.run()

    saveModelTrainingHistory = SaveModelTrainingExperiment(builtModel, modelConfig, modelTrainingHistory)
    saveModelTrainingHistory.run()
