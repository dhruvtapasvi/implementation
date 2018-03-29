from keras import Model
from abc import ABCMeta, abstractclassmethod


class Autoencoder(metaclass=ABCMeta):
    @abstractclassmethod
    def encoder(self) -> Model:
        raise NotImplementedError

    @abstractclassmethod
    def decoder(self) -> Model:
        raise NotImplementedError

    @abstractclassmethod
    def autoencoder(self) -> Model:
        raise NotImplementedError
