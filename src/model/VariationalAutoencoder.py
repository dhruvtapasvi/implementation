from abc import ABCMeta, abstractclassmethod
import numpy as np
from keras.models import Model


class VariationalAutoencoder(metaclass=ABCMeta):
    @abstractclassmethod
    def summary(self):
        raise NotImplementedError

    @abstractclassmethod
    def train(
            self,
            trainingData: np.ndarray,
            validationData: np.ndarray,
            epochs,
            batchSize):
        raise NotImplementedError

    @abstractclassmethod
    def autoencoder(self) -> Model:
        raise NotImplementedError

    @abstractclassmethod
    def encoder(self) -> Model:
        raise NotImplementedError

    @abstractclassmethod
    def decoder(self) -> Model:
        raise NotImplementedError
