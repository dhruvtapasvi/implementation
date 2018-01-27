from abc import ABCMeta, abstractclassmethod
import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model

from model.sampling import samplingConstructor
from model.variationalAutoencoderLoss import variationalAutoencoderLoss
from model.AlreadyTrainedError import AlreadyTrainedError


class VariationalAutoencoder(metaclass=ABCMeta):
    def __init__(self, inputRepresentationDimensions, latentRepresentationDimension):
        self.__inputRepresentationDimensions = inputRepresentationDimensions
        self.__latentRepresentationDimension = latentRepresentationDimension
        self.__isTrained = False

    def buildModels(self):
        encoderLayers = self.encoderLayersConstructor()
        decoderLayers = self.decoderLayersConstructor()

        self.__buildAutoencoder(encoderLayers, decoderLayers)
        self.__buildEncoder(encoderLayers)
        self.__buildDecoder(decoderLayers)

    def __buildAutoencoder(self, encoderLayers, decoderLayers):
        # Input to the encoder and autoencoder models:
        inputRepresentation = Input(shape=self.__inputRepresentationDimensions)

        latentRepresentationMean, latentRepresentationLogVariance = encoderLayers(inputRepresentation)
        latentRepresentation = Lambda(
            samplingConstructor(self.__latentRepresentationDimension),
            output_shape=(self.__latentRepresentationDimension,)
        )([latentRepresentationMean, latentRepresentationLogVariance])

        decodedInputRepresentation = decoderLayers(latentRepresentation)

        self.__autoencoder = Model(inputRepresentation, decodedInputRepresentation)

        self.__autoencoder.add_loss(variationalAutoencoderLoss(
            self.__inputRepresentationDimensions,
            inputRepresentation,
            decodedInputRepresentation,
            latentRepresentationMean,
            latentRepresentationLogVariance
        ))
        self.__autoencoder.compile(optimizer='rmsprop', loss=None)

    def __buildEncoder(self, encoderLayers):
        inputRepresentation = Input(shape=self.__inputRepresentationDimensions)
        latentRepresentationMean, _ = encoderLayers(inputRepresentation)
        self._encoder = Model(inputRepresentation, latentRepresentationMean)

    def __buildDecoder(self, decoderLayers):
        customLatentRepresentation = Input(shape=(self.__latentRepresentationDimension,))
        customDecodedInputRepresentation = decoderLayers(customLatentRepresentation)
        self.__decoder = Model(customLatentRepresentation, customDecodedInputRepresentation)

    @abstractclassmethod
    def encoderLayersConstructor(self):
        raise NotImplementedError

    @abstractclassmethod
    def decoderLayersConstructor(self):
        raise NotImplementedError

    def encoder(self) -> Model:
        return self._encoder

    def decoder(self) -> Model:
        return self.__decoder

    def autoencoder(self) -> Model:
        return self.__autoencoder

    def train(
            self,
            trainingData: np.ndarray,
            validationData: np.ndarray,
            epochs,
            batchSize):
        if self.__isTrained:
            raise AlreadyTrainedError
        else:
            self.__autoencoder.fit(
                trainingData,
                shuffle=True,
                epochs=epochs,
                batch_size=batchSize,
                validation_data=(validationData, None))
            self.__isTrained = True

    def summary(self):
        self.__autoencoder.summary()
