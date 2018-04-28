import numpy as np

from dataset.loader.DatasetLoader import DatasetLoader
from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.interpolate.process.CreateTransformedInterpolateData import CreateTransformedInterpolateData
import dataset.info.MnistTransformedInfoLimitedRotation as mnistTransformedLimitedRotationInfo


class MnistTransformedLimitedRotationInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, mnistLoader: DatasetLoader):
        self.__mnistLoader = CreateTransformedInterpolateData(
            mnistLoader,
            mnistTransformedLimitedRotationInfo.PADDING,
            *mnistTransformedLimitedRotationInfo.DEFAULT_JOINT_FACTORS,
            *mnistTransformedLimitedRotationInfo.INTERPOLATE_JOINT_FACTORS
        )

    def loadInterpolationData(self):
        return self.__mnistLoader.loadInterpolationData()

    def dataPointShape(self):
        return mnistTransformedLimitedRotationInfo.IMAGE_DIMENSIONS, mnistTransformedLimitedRotationInfo.LABEL_DIMENSIONS
