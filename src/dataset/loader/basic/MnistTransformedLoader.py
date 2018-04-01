import numpy as np

import dataset.info.MnistTransformedInfo as mnistTransformedInfo
from dataset.loader.DatasetLoader import DatasetLoader


class MnistTransformedLoader(DatasetLoader):
    def __init__(self, transformedMnistHome: str):
        self.__transformedMnistHome = transformedMnistHome

    def loadData(self):
        xTrain = np.load(self.__transformedMnistHome + "/x_train.npy")
        yTrain = np.load(self.__transformedMnistHome + "/y_train.npy")
        xVal = np.load(self.__transformedMnistHome + "/x_val.npy")
        yVal = np.load(self.__transformedMnistHome + "/y_val.npy")
        xTest = np.load(self.__transformedMnistHome + "/x_test.npy")
        yTest = np.load(self.__transformedMnistHome + "/y_test.npy")
        return (xTrain, yTrain), (xVal, yVal), (xTest, yTest)

    def dataPointShape(self):
        return mnistTransformedInfo.IMAGE_DIMENSIONS, mnistTransformedInfo.LABEL_DIMENSIONS
