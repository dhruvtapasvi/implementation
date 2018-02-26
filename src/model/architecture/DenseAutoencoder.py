import numpy as np
from keras.layers import Dense, Flatten, Reshape

from model.VariationalAutoencoder import VariationalAutoencoder


class DenseAutoencoder(VariationalAutoencoder):
    # Architecture from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    def __init__(self, reconstructionLossConstructor, klLossWeight, inputRepresentationDimensions, intermediateRepresentationDimension, latentRepresentationDimension):
        super().__init__(reconstructionLossConstructor, klLossWeight, inputRepresentationDimensions, latentRepresentationDimension)
        self.__inputRepresentationDimensions = inputRepresentationDimensions
        self.__intermediateRepresentationDimension = intermediateRepresentationDimension
        self.__latentRepresentationDimension = latentRepresentationDimension

    def encoderLayersConstructor(self):
        inputToFlattenedInput = Flatten()
        flattenedInputToIntermediate = Dense(self.__intermediateRepresentationDimension, activation='relu')
        intermediateToLatentMean = Dense(self.__latentRepresentationDimension)
        intermediateToLatentLogVariance = Dense(self.__latentRepresentationDimension)

        def encoderLayers(inputRepresentation):
            flattenedInputRepresentation = inputToFlattenedInput(inputRepresentation)
            intermediateRepresentation = flattenedInputToIntermediate(flattenedInputRepresentation)
            latentRepresentationMean = intermediateToLatentMean(intermediateRepresentation)
            latentRepresentationLogVariance = intermediateToLatentLogVariance(intermediateRepresentation)
            return latentRepresentationMean, latentRepresentationLogVariance

        return encoderLayers

    def decoderLayersConstructor(self):
        totalNumberOfPixels = np.prod(self.__inputRepresentationDimensions)
        latentToIntermediate = Dense(self.__intermediateRepresentationDimension, activation='relu')
        intermediateToFlattenedInput = Dense(totalNumberOfPixels, activation='sigmoid')
        flattenedInputToInput = Reshape(self.__inputRepresentationDimensions)

        def decoderLayers(latentRepresentation):
            decodedIntermediateRepresentation = latentToIntermediate(latentRepresentation)
            decodedFlattenedInputRepresentation = intermediateToFlattenedInput(decodedIntermediateRepresentation)
            decodedInputRepresentation = flattenedInputToInput(decodedFlattenedInputRepresentation)
            return decodedInputRepresentation

        return decoderLayers
