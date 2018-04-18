from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
from config.DenseAutoencoderConfig import DenseAutoencoderConfig
from config import routes

from dataset.info import MnistInfo as mnistInfo, MnistTransformedInfo as mnistTranformedInfo, NorbInfo as norbInfo, ShapesInfo as shapesInfo

from dataset.loader.basic.MnistLoader import MnistLoader
from dataset.loader.basic.NorbLoader import NorbLoader
from dataset.loader.basic.LoadFromFile import LoadFromFile
from dataset.loader.preprocess.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from dataset.loader.preprocess.Pad import Pad

from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.TrainModelExperiment import TrainModelExperiment
from experiment.SaveModelTrainingExperiment import SaveModelTrainingExperiment


resourcesRoot = routes.RESOURCE_ROUTE
mnistLoader = ScaleBetweenZeroAndOne(MnistLoader(), *mnistInfo.MNIST_RANGE)
norbLoader = ScaleBetweenZeroAndOne(NorbLoader(resourcesRoot + "/norb"), *norbInfo.NORB_RANGE)
mnistTransformedLoader = ScaleBetweenZeroAndOne(LoadFromFile(resourcesRoot + "/mnistTransformed_10", mnistTranformedInfo.IMAGE_DIMENSIONS, mnistTranformedInfo.LABEL_DIMENSIONS), *mnistTranformedInfo.RANGE)
shapesLoader = ScaleBetweenZeroAndOne(LoadFromFile(routes.RESOURCE_ROUTE + shapesInfo.HOME, shapesInfo.BASE_IMAGE_SIZE, shapesInfo.BASE_IMAGE_SIZE), *shapesInfo.RANGE)
paddedMnistLoader = ScaleBetweenZeroAndOne(Pad(MnistLoader(), ((18, 18), (18, 18)), (64, 64), (1,)), *mnistInfo.MNIST_RANGE)

modelConfigRoot = routes.CONFIG_ROUTE + "/model"
modelConfigs = [
    # (modelConfig, DatasetLoader, epochs, batchSize)
    (ConvolutionalAutoencoderConfig(modelConfigRoot + "/convolutional/mnist_padded_conv_7_16_256_32_bce.json"), paddedMnistLoader, 100, 1000)
]

for modelConfig, datasetLoader, epochs, batchSize in modelConfigs:
    buildModel = BuildModelExperiment(modelConfig)
    builtModel = buildModel.run()
    
    trainModel = TrainModelExperiment(builtModel, datasetLoader, epochs, batchSize)
    modelTrainingHistory = trainModel.run()

    saveModelTrainingHistory = SaveModelTrainingExperiment(builtModel, modelConfig, modelTrainingHistory)
    saveModelTrainingHistory.run()
