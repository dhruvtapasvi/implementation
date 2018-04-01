import numpy as np

from dataset.interpolateLoader.InterpolateDatasetLoader import InterpolateDatasetLoader
import dataset.info.MnistTransformedInfo as mnistTransformedInfo


class MnistTransformedInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, mnistTransformedInterpolateHome: str):
        self.__mnistTransformedInterpolateHome = mnistTransformedInterpolateHome

    def loadInterpolationData(self):
        xTestInterpolate = np.load(self.__mnistTransformedInterpolateHome + "/xInterpolate.npy")
        yTestInterpolate = np.load(self.__mnistTransformedInterpolateHome + "/yInterpolate.npy")
        return xTestInterpolate, yTestInterpolate

    def dataPointShape(self):
        return mnistTransformedInfo.IMAGE_DIMENSIONS, mnistTransformedInfo.LABEL_DIMENSIONS
