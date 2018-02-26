from datasets.DatasetLoader import DatasetLoader
import numpy as np


class ScaleBetweenZeroAndOne(DatasetLoader):
    def __init__(self, baseDatasetLoader: DatasetLoader, minimum, maximum):
        self.__baseDatasetLoader = baseDatasetLoader
        self.__minimum = float(minimum)
        self.__maxrange = float(maximum - minimum)

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (XTrain, YTrain), (XValidation, YValidation), (XTest, YTest) = self.__baseDatasetLoader.loadData()
        XTrain = self.__scaleAndAdjustMinimum(XTrain)
        XValidation = self.__scaleAndAdjustMinimum(XValidation)
        XTest = self.__scaleAndAdjustMinimum(XTest)
        return (XTrain, YTrain), (XValidation, YValidation), (XTest, YTest)

    def __scaleAndAdjustMinimum(self, unscaled: np.ndarray):
        return unscaled.astype('float64') / self.__maxrange + self.__minimum
