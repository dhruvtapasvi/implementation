import numpy as np


import dataset.info.MnistTransformedInfo as mnistTransformedInfo
from dataset.interpolateLoader.process.CreateTransformedInterpolateData import CreateTransformedInterpolateData
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
        print(interpolateTest.shape)
        print(interpolateTestLabels.shape)

        startIndex = 20000
        numElements = 10
        # print(interpolateTestLabels[startIndex:startIndex + numElements])
        print("Display")
        imagesArrayComparisonDisplay(np.swapaxes(interpolateTest, 0, 1), "./out2/mnistTransformCreateInterpolationData.png", startIndex=startIndex, endIndex=startIndex+numElements)
