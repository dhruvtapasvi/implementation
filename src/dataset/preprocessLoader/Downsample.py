from dataset.DatasetLoader import DatasetLoader
import numpy as np
from skimage.transform import resize


class Downsample(DatasetLoader):
    def __init__(self, baseDatasetLoader: DatasetLoader, firstDimSize, secondDimSize):
        self.__baseDatasetLoader = baseDatasetLoader
        self.__firstDimSize = firstDimSize
        self.__secondDimSize = secondDimSize

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (XTrain, YTrain), (XValidation, YValidation), (XTest, YTest) = self.__baseDatasetLoader.loadData()
        XTrain = self.__resize(XTrain)
        XValidation = self.__resize(XValidation)
        XTest = self.__resize(XTest)
        return (XTrain, YTrain), (XValidation, YValidation), (XTest, YTest)

    def __resize(self, images: np.ndarray):
        return np.array([resize(image, (self.__firstDimSize, self.__secondDimSize)) for image in images])
