from skimage.util import pad
import numpy as np

from dataset.loader.DatasetLoader import DatasetLoader


class Pad(DatasetLoader):
    def __init__(self, baseDatasetLoader, padding, imageDimensions, labelDimensions):
        self.__baseDatasetLoader = baseDatasetLoader
        self.__padding = padding
        self.__imageDimensions = imageDimensions
        self.__labelDimensions = labelDimensions

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (xTrain, yTrain), (xVal, yVal), (xTest, yTest) = self.__baseDatasetLoader.loadData()
        xTrainPadded = self.__pad(xTrain)
        xValPadded = self.__pad(xVal)
        xTestPadded = self.__pad(xTrain)
        return (xTrainPadded, yTrain), (xValPadded, yVal), (xTestPadded, yTest)

    def __pad(self, images):
        return np.array([pad(image, self.__padding, 'constant') for image in images])

    def dataPointShape(self):
        return self.__imageDimensions, self.__labelDimensions
