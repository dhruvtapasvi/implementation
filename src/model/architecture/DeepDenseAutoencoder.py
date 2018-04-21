import numpy as np
from keras.layers import Dense, Flatten, Reshape, BatchNormalization

from model.VariationalAutoencoder import VariationalAutoencoder


class DeepDenseAutoencoder(VariationalAutoencoder):
    def __init__(self, reconstructionLossConstructor, klLossWeight, inputDimensions, encoderIntermediateDimensions, decoderIntermediateDimensions, latentDimension):
        super().__init__(reconstructionLossConstructor, klLossWeight, inputDimensions, latentDimension)
        self.__inputDimensions = inputDimensions
        self.__encoderIntermediateDimensions = encoderIntermediateDimensions
        self.__decoderIntermediateDimensions = decoderIntermediateDimensions
        self.__latentDimension = latentDimension

    def encoderLayersConstructor(self):
        layers = [
            Flatten(),
            BatchNormalization()
        ] + [
            self.__denseLayerConstructor(size) for size in self.__encoderIntermediateDimensions
        ]
        intermediateToLatentMean = Dense(self.__latentDimension)
        intermediateToLatentLogVariance = Dense(self.__latentDimension)

        def encoderLayers(input):
            intermediate = self.evaluateLayersList(layers, input)
            latentMean = intermediateToLatentMean(intermediate)
            latentLogVariance = intermediateToLatentLogVariance(intermediate)
            return latentMean, latentLogVariance

        return encoderLayers

    def decoderLayersConstructor(self):
        totalNumberOfPixels = np.prod(self.__inputDimensions)
        layers = [
            BatchNormalization()
        ] + [
            self.__denseLayerConstructor(size) for size in self.__decoderIntermediateDimensions
        ] + [
            Dense(totalNumberOfPixels, activation='sigmoid', kernel_initializer='glorot_normal'),
            Reshape(self.__inputDimensions)
        ]
        return self.collapseLayers(layers)

    def __denseLayerConstructor(self, size):
        layers = [
            Dense(size, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization()
        ]
        return self.collapseLayers(layers)
