from datasets.DatasetLoader import DatasetLoader
import numpy as np
from keras.datasets import mnist


class MnistLoader(DatasetLoader):
    __MNIST_VALIDATION_SPLIT = 50000

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (XTest, YTest), (XTrain, YTrain) = mnist.load_data()
        return\
            (XTrain[:MnistLoader.__MNIST_VALIDATION_SPLIT], YTrain[:MnistLoader.__MNIST_VALIDATION_SPLIT]),\
            (XTrain[MnistLoader.__MNIST_VALIDATION_SPLIT:], YTrain[MnistLoader.__MNIST_VALIDATION_SPLIT:]),\
            (XTest, YTest)
