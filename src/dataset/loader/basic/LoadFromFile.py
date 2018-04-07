import numpy as np

from dataset.loader.DatasetLoader import DatasetLoader


class LoadFromFile(DatasetLoader):
    def __init__(self, home: str, xDimensions, yDimensions):
        self.__home = home
        self.__dataPointShape = (xDimensions, yDimensions)

    def loadData(self):
        xTrain = np.load(self.__home + "/x_train.npy")
        yTrain = np.load(self.__home + "/y_train.npy")
        xVal = np.load(self.__home + "/x_val.npy")
        yVal = np.load(self.__home + "/y_val.npy")
        xTest = np.load(self.__home + "/x_test.npy")
        yTest = np.load(self.__home + "/y_test.npy")
        return (xTrain, yTrain), (xVal, yVal), (xTest, yTest)

    def dataPointShape(self):
        return self.__dataPointShape
