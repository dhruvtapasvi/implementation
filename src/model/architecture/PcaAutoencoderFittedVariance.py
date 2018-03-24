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
        # inputToFlattenedInput = Flatten()
        inputBatchNormalisation = BatchNormalization()
        encoderLayer = self.__compoundLayerConstructor()
        intermediateToLatentMean = Dense(self.__latentRepresentationDimension)
        intermediateToLatentLogVariance = Dense(self.__latentRepresentationDimension)

        def encoderLayers(inputRepresentation):
            # intermediateRepresentation = inputToFlattenedInput(inputRepresentation)
            intermediateRepresentation = inputBatchNormalisation(inputRepresentation)
            for i in range(self.__depth):
                intermediateRepresentation = encoderLayer(intermediateRepresentation)
            latentRepresentationMean = intermediateToLatentMean(intermediateRepresentation)
            latentRepresentationVariance = intermediateToLatentLogVariance(intermediateRepresentation)
            return latentRepresentationMean, latentRepresentationVariance

        return encoderLayers

    def decoderLayersConstructor(self):
        totalNumberOfPixels = np.prod(self.__inputRepresentationDimensions)
        decoderLayer = self.__compoundLayerConstructor()
        intermediateToFlattenedInput = Dense(totalNumberOfPixels)
        flattenedInputToInput = Reshape(self.__inputRepresentationDimensions)
        intermediateToFlattenedInputVariance = Dense(totalNumberOfPixels,activation="relu")
        flattenedInputVarianceToInputVariance = Reshape(self.__inputRepresentationDimensions)

        def decoderLayers(latentRepresentation):
            intermediateRepresentation = latentRepresentation
            for i in range(self.__depth):
                intermediateRepresentation = decoderLayer(intermediateRepresentation)
            decodedFlattenedInputRepresentation = intermediateToFlattenedInput(intermediateRepresentation)
            decodedInputRepresentation = flattenedInputToInput(decodedFlattenedInputRepresentation)
            decodedFlattenedInputRepresentationVariance = intermediateToFlattenedInputVariance(intermediateRepresentation)
            decodedInputRepresentationVariance = flattenedInputVarianceToInputVariance(decodedFlattenedInputRepresentationVariance)
            return decodedInputRepresentation, decodedInputRepresentationVariance

        return decoderLayers


    def __compoundLayerConstructor(self):
        def compoundLayer(compoundLayerInput):
            denseResult = (Dense(self.__intermediateRepresentationDimension))(compoundLayerInput)
            batchNormalisationResult = (BatchNormalization())(denseResult)
            activationResult = (Activation("relu"))(batchNormalisationResult)
            dropoutResult = (Dropout(self.__dropout))(activationResult)
            return dropoutResult

        return compoundLayer
