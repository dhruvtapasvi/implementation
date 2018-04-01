import numpy as np
import os


import dataset.info.MnistTransformedInfo as mnistTransformedInfo
from dataset.interpolate.process.CreateTransformedInterpolateData import CreateTransformedInterpolateData
from dataset.loader.basic.MnistLoader import MnistLoader
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
from experiment.Experiment import Experiment


class CreateMnistTransformedInterpolationDatasetExperiment(Experiment):
    def run(self):
        mnistLoader = MnistLoader()
        mnistSize = mnistLoader.dataPointShape()[0]
        mnistTransformedSize = mnistTransformedInfo.IMAGE_DIMENSIONS

        firstAxisPadding = (mnistTransformedSize[0] - mnistSize[0]) // 2
        secondAxisPadding = (mnistTransformedSize[1] - mnistSize[1]) // 2
        padding = ((firstAxisPadding, firstAxisPadding), (secondAxisPadding, secondAxisPadding))

        mnistTransformedInterpolatedLoader = CreateTransformedInterpolateData(
            MnistLoader(),
            padding,
            mnistTransformedInfo.INTERPOLATE_ROTATION_FACTORS,
            (-mnistTransformedInfo.TRANSFORM_SHEAR_FACTOR, mnistTransformedInfo.TRANSFORM_SHEAR_FACTOR, 0.0, mnistTransformedInfo.INTERPOLATE_INCORRECT_SHEAR_FACTOR),
            (-mnistTransformedInfo.TRANSFORM_LOG2_STRETCH_FACTOR, mnistTransformedInfo.TRANSFORM_LOG2_STRETCH_FACTOR, 0.0, mnistTransformedInfo.INTERPOLATE_INCORRECT_LOG_2_STRETCH_FACTOR)
        )
        interpolateTest, interpolateTestLabels = mnistTransformedInterpolatedLoader.loadInterpolationData()

        rootFolder = "."
        datasetRoute = rootFolder + "/res/mnistTransformedInterpolate"

        if not os.path.isdir(datasetRoute):
            os.mkdir(datasetRoute)
            np.save(datasetRoute + "/xInterpolate.npy", interpolateTest)
            np.save(datasetRoute + "/yInterpolate.npy", interpolateTestLabels)

        print(interpolateTest.shape)
        print(interpolateTestLabels.shape)
