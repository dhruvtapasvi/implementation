from datasets.DatasetLoader import DatasetLoader
import numpy as np
from keras.datasets import mnist


class MnistLoader(DatasetLoader):
    __MNIST_VALIDATION_SPLIT = 50000

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (XTrain, YTrain), test = mnist.load_data()
        train = XTrain[:MnistLoader.__MNIST_VALIDATION_SPLIT], YTrain[:MnistLoader.__MNIST_VALIDATION_SPLIT]
        validation = XTrain[MnistLoader.__MNIST_VALIDATION_SPLIT:], YTrain[MnistLoader.__MNIST_VALIDATION_SPLIT:]
        return train, validation, test
