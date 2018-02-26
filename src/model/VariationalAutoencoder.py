from abc import ABCMeta, abstractclassmethod

import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model

from model.AlreadyTrainedError import AlreadyTrainedError
from model.loss.kullbackLeiberLoss import kullbackLeiberLossConstructor
from model.loss.binaryCrossEntropyLoss import binaryCrossEntropyLossConstructor
from model.loss.variationalAutoencoderLoss import variationalAutoencoderLossConstructor
from model.sampling import samplingConstructor


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

        self.__autoencoder.compile(
            optimizer='rmsprop',
            loss=variationalAutoencoderLossConstructor(self.__inputRepresentationDimensions, latentRepresentationMean, latentRepresentationLogVariance),
            metrics=[
                binaryCrossEntropyLossConstructor(self.__inputRepresentationDimensions),
                kullbackLeiberLossConstructor(latentRepresentationMean, latentRepresentationLogVariance)
            ]
        )

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

    def evaluateLayersList(self, layersList, input):
        intermediateResult = input
        for layer in layersList:
            intermediateResult = layer(intermediateResult)
        return intermediateResult

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
                trainingData,
                shuffle=True,
                epochs=epochs,
                batch_size=batchSize,
                validation_data=(validationData, validationData))
            self.__isTrained = True

    def summary(self):
        self.__autoencoder.summary()

    def saveWeights(self, location):
        self.__autoencoder.save_weights(location)

    def loadWeights(self, location):
        self.__autoencoder.load_weights(location)
