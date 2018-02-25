from datasets.DatasetLoader import DatasetLoader
import numpy as np
from keras.datasets import mnist


class MnistLoader(DatasetLoader):
    __MNIST_VALIDATION_SPLIT = 50000

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (XTest, YTest), (XTrain, YTrain) = mnist.load_data()
        return\
            (XTrain[:self.__MNIST_VALIDATION_SPLIT], YTrain[:self.__MNIST_VALIDATION_SPLIT]),\
            (XTrain[self.__MNIST_VALIDATION_SPLIT:], YTrain[self.__MNIST_VALIDATION_SPLIT:]),\
            (XTest, YTest)
