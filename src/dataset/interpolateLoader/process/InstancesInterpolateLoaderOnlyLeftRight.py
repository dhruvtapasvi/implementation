from dataset.interpolateLoader.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.loader.DatasetLoader import DatasetLoader
import numpy as np


class InstancesInterpolateLoaderOnlyLeftRight(InterpolateDatasetLoader):
    # Hand pick instances from the test dataset
    # Only return the left and right values, not the control positive and negative values

    def __init__(self, baseDatasetLoader: DatasetLoader, instancesToLoad: np.ndarray):
        #  instances contains pairs of instances from the test dataset
        self.__baseDatasetLoader = baseDatasetLoader
        self.__instances = instancesToLoad

    def loadInterpolationData(self):
        _, _, (xTest, yTest) = self.__baseDatasetLoader.loadData()
        xLeftRight = xTest[self.__instances]
        yLeftRight = yTest[self.__instances]
        return xLeftRight, yLeftRight
