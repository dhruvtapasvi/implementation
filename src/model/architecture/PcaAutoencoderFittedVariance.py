import numpy as np
from keras.layers import Dense, Flatten, Reshape, BatchNormalization, Activation, Dropout

from model.VariationalAutoencoderFittedVariance import VariationalAutoencoderFittedVariance


class PcaAutoencoderFittedVariance(VariationalAutoencoderFittedVariance):
    def __init__(self, reconstructionLossConstructor, klLossWeight, inputRepresentationDimensions, intermediateRepresentationDimension, numIntermediateDimensions, latentRepresentationDimension, dropout):
        super().__init__(reconstructionLossConstructor, klLossWeight, inputRepresentationDimensions, latentRepresentationDimension)
        self.__inputRepresentationDimensions = inputRepresentationDimensions
        self.__intermediateRepresentationDimension = intermediateRepresentationDimension
        self.__depth = numIntermediateDimensions
        self.__latentRepresentationDimension = latentRepresentationDimension
        self.__dropout = dropout

    def encoderLayersConstructor(self):
        inputBatchNormalisation = BatchNormalization()
        encoderLayersList = [self.__compoundLayerConstructor() for _ in range(self.__depth)]
        intermediateToLatentMean = Dense(self.__latentRepresentationDimension)
        intermediateToLatentLogVariance = Dense(self.__latentRepresentationDimension)

        def encoderLayers(inputRepresentation):
            intermediateRepresentation = inputBatchNormalisation(inputRepresentation)
            intermediateRepresentation = self.evaluateLayersList(encoderLayersList, intermediateRepresentation)
            latentRepresentationMean = intermediateToLatentMean(intermediateRepresentation)
            latentRepresentationVariance = intermediateToLatentLogVariance(intermediateRepresentation)
            return latentRepresentationMean, latentRepresentationVariance

        return encoderLayers

    def decoderLayersConstructor(self):
        totalNumberOfPixels = np.prod(self.__inputRepresentationDimensions)
        decoderLayersList = [self.__compoundLayerConstructor() for _ in range(self.__depth)]
        intermediateToFlattenedInput = Dense(totalNumberOfPixels)
        flattenedInputToInput = Reshape(self.__inputRepresentationDimensions)
        intermediateToFlattenedInputVariance = Dense(totalNumberOfPixels,activation="relu", kernel_initializer="he_normal", bias_initializer="uniform")
        flattenedInputVarianceToInputVariance = Reshape(self.__inputRepresentationDimensions)

        def decoderLayers(latentRepresentation):
            intermediateRepresentation = latentRepresentation
            intermediateRepresentation = self.evaluateLayersList(decoderLayersList, intermediateRepresentation)
            decodedFlattenedInputRepresentation = intermediateToFlattenedInput(intermediateRepresentation)
            decodedInputRepresentation = flattenedInputToInput(decodedFlattenedInputRepresentation)
            decodedFlattenedInputRepresentationVariance = intermediateToFlattenedInputVariance(intermediateRepresentation)
            decodedInputRepresentationVariance = flattenedInputVarianceToInputVariance(decodedFlattenedInputRepresentationVariance)
            return decodedInputRepresentation, decodedInputRepresentationVariance

        return decoderLayers

    def __compoundLayerConstructor(self):
        denseLayer = Dense(self.__intermediateRepresentationDimension, activation="relu", kernel_initializer="he_normal", bias_initializer="uniform")
        batchNormalisationLayer = BatchNormalization()
        dropoutLayer = Dropout(self.__dropout)

        def compoundLayer(compoundLayerInput):
            return dropoutLayer(batchNormalisationLayer(denseLayer(compoundLayerInput)))

        return compoundLayer
