from dataset.DatasetLoader import DatasetLoader
import numpy as np
from keras.datasets import mnist
import dataset.info.MnistInfo as mnistInfo


class MnistLoader(DatasetLoader):
    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (XTrain, YTrain), (XTest, YTest) = mnist.load_data()
        YTrain = np.array([[i] for i in YTrain])
        YTest = np.array([[i] for i in YTest])
        train = XTrain[:mnistInfo.MNIST_VALIDATION_SPLIT], YTrain[:mnistInfo.MNIST_VALIDATION_SPLIT]
        validation = XTrain[mnistInfo.MNIST_VALIDATION_SPLIT:], YTrain[mnistInfo.MNIST_VALIDATION_SPLIT:]
        return train, validation, (XTest, YTest)

    def dataPointShape(self):
        return mnistInfo.MNIST_IMAGE_DIMENSIONS, mnistInfo.MNIST_LABEL_DIMENSIONS
