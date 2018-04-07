from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
from config.DenseAutoencoderConfig import DenseAutoencoderConfig
from config import routes

import dataset.info.MnistInfo as mnistInfo
import dataset.info.NorbInfo as norbInfo
import dataset.info.MnistTransformedInfo as mnistTransformedInfo
import dataset.info.ShapesInfo as shapesInfo

from dataset.loader.basic.MnistLoader import MnistLoader
from dataset.loader.basic.NorbLoader import NorbLoader
from dataset.loader.basic.LoadFromFile import LoadFromFile
from dataset.loader.preprocess.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne

from dataset.interpolate.basic.MnistInterpolateLoader import MnistInterpolateLoader
from dataset.interpolate.basic.NorbInterpolateLoader import NorbInterpolateLoader
from dataset.interpolate.basic.MnistTransformedInterpolateLoader import MnistTransformedInterpolateLoader
from dataset.interpolate.process.ScaleBetweenZeroAndOneInterpolate import ScaleBetweenZeroAndOneInterpolate

from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.LoadModelExperiment import LoadModelExperiment
from experiment.ReconstructionsExperiment import ReconstructionsExperiment
from experiment.InterpolateExperiment import InterpolateExperiment


mnistLoader = MnistLoader()
norbLoader = NorbLoader(norbInfo.NORB_HOME)
mnistTransformedLoader = LoadFromFile(mnistTransformedInfo.MNIST_TRANSFORMED_10_HOME, mnistTransformedInfo.IMAGE_DIMENSIONS, mnistTransformedInfo.LABEL_DIMENSIONS)
shapesLoader = ScaleBetweenZeroAndOne(LoadFromFile(routes.RESOURCE_ROUTE + shapesInfo.HOME, shapesInfo.BASE_IMAGE_SIZE, shapesInfo.BASE_IMAGE_SIZE), *shapesInfo.RANGE)

mnistLoaderScaled = ScaleBetweenZeroAndOne(mnistLoader, *mnistInfo.MNIST_RANGE)
norbLoaderScaled = ScaleBetweenZeroAndOne(norbLoader, *norbInfo.NORB_RANGE)
mnistTransformedLoaderScaled = ScaleBetweenZeroAndOne(mnistTransformedLoader, *mnistTransformedInfo.RANGE)

mnistInterpolateLoader = ScaleBetweenZeroAndOneInterpolate(MnistInterpolateLoader(mnistLoader), *mnistInfo.MNIST_RANGE)
norbInterpolateLoader = ScaleBetweenZeroAndOneInterpolate(NorbInterpolateLoader(norbLoader), *norbInfo.NORB_RANGE)
mnistTransformedInterpolateLoader = ScaleBetweenZeroAndOneInterpolate(MnistTransformedInterpolateLoader(mnistLoader), *mnistTransformedInfo.RANGE)

configDatasetTuples = [
    (ConvolutionalAutoencoderConfig(routes.getConfigRoute("model/convolutional/mnist_transformed_conv_7_16_256_32_bce.json")), shapesLoader, mnistTransformedInterpolateLoader)
]

for config, loader, interpolateLoader in configDatasetTuples:
    variationalAutoencoderBuilder = BuildModelExperiment(config)
    variationalAutoencoder = variationalAutoencoderBuilder.run()

    loadWeights = LoadModelExperiment(config, variationalAutoencoder)
    loadWeights.run()

    createReconstructions = ReconstructionsExperiment(loader, config, variationalAutoencoder, 100, 10)
    createReconstructions.run()

    # interpolate = InterpolateExperiment(interpolateLoader)
    # interpolate.run()
