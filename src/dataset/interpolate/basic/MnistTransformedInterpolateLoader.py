import numpy as np

from dataset.loader.DatasetLoader import DatasetLoader
from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.interpolate.process.CreateTransformedInterpolateData import CreateTransformedInterpolateData
import dataset.info.MnistTransformedInfo as mnistTransformedInfo


class MnistTransformedInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, mnistLoader: DatasetLoader):
        self.__mnistLoader = CreateTransformedInterpolateData(
            mnistLoader,
            mnistTransformedInfo.PADDING,
            *mnistTransformedInfo.DEFAULT_JOINT_FACTORS,
            *mnistTransformedInfo.INTERPOLATE_JOINT_FACTORS
        )

    def loadInterpolationData(self):
        return self.__mnistLoader.loadInterpolationData()

    def dataPointShape(self):
        return mnistTransformedInfo.IMAGE_DIMENSIONS, mnistTransformedInfo.LABEL_DIMENSIONS
