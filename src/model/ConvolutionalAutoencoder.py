import numpy as np
from keras.layers import Dense, Flatten, Reshape, Conv2D, Deconv2D

from model.VariationalAutoencoder import VariationalAutoencoder


class ConvolutionalAutoencoder(VariationalAutoencoder):
    # Architecture from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    def __init__(self, inputRepresentationDimensions, numberConvolutions, baseConvolutionalDepth, intermediateRepresentationDimension, latentRepresentationDimension):
        super().__init__(inputRepresentationDimensions, latentRepresentationDimension)
        self.__baseConvolutionalDepth = baseConvolutionalDepth
        self.__numberConvolutions = numberConvolutions
        self.__inputRepresentationDimensions = inputRepresentationDimensions
        self.__intermediateRepresentationDimension = intermediateRepresentationDimension
        self.__latentRepresentationDimension = latentRepresentationDimension

    def encoderLayersConstructor(self):
        convolutionalShape = self.__inputRepresentationDimensions + (1,)
        encoderLayersList = [
            Reshape(convolutionalShape),
            self.__buildConvolutionalsAndDownSamplings(),
            Flatten(),
            Dense(self.__intermediateRepresentationDimension, activation='relu')
        ]
        intermediateToLatentMean = Dense(self.__latentRepresentationDimension)
        intermediateToLatentLogVariance = Dense(self.__latentRepresentationDimension)

        def encoderLayers(inputRepresentation):
            intermediateRepresentation = self.evaluateLayersList(encoderLayersList, inputRepresentation)
            latentRepresentationMean = intermediateToLatentMean(intermediateRepresentation)
            latentRepresentationLogVariance = intermediateToLatentLogVariance(intermediateRepresentation)
            return latentRepresentationMean, latentRepresentationLogVariance

        return encoderLayers

    def decoderLayersConstructor(self):
        shrinkFactor = 2 ** (self.__numberConvolutions - 1)
        convolutionalTransposeDimensions = tuple(map(lambda x: x // shrinkFactor, self.__inputRepresentationDimensions)) + (shrinkFactor * self.__baseConvolutionalDepth,)
        print(convolutionalTransposeDimensions)
        totalNumberOfNodes = np.prod(convolutionalTransposeDimensions)

        decoderLayersList = [
            Dense(self.__intermediateRepresentationDimension, activation='relu'),
            Dense(totalNumberOfNodes, activation='relu'),
            Reshape(convolutionalTransposeDimensions),
            self.__buildConvolutionalTransposesAndUpsampling(),
            Reshape(self.__inputRepresentationDimensions)
        ]

        def decoderLayers(latentRepresentation):
            return self.evaluateLayersList(decoderLayersList, latentRepresentation)

        return decoderLayers

    def __buildConvolutionalsAndDownSamplings(self):
        convolutionalDepth = self.__baseConvolutionalDepth
        convolutionalLayers = []
        for i in range(self.__numberConvolutions):
            for j in range(2):
                convolutionalLayers.append(Conv2D(convolutionalDepth, (3, 3), activation='relu', padding='same'))
            if i < self.__numberConvolutions - 1:
                convolutionalLayers.append(Conv2D(convolutionalDepth, (2, 2), strides=(2, 2), activation='relu', padding='same'))
                convolutionalDepth *= 2

        def convolutionsAndDownsampling(preconvolutionalRepresentation):
            return self.evaluateLayersList(convolutionalLayers, preconvolutionalRepresentation)

        return convolutionsAndDownsampling

    def __buildConvolutionalTransposesAndUpsampling(self):
        convolutionalTransposeDepth = self.__baseConvolutionalDepth * (2 ** (self.__numberConvolutions - 1))
        convolutionalTransposeLayers = []
        for i in range(self.__numberConvolutions):
            if i > 0:
                convolutionalTransposeLayers.append(Deconv2D(convolutionalTransposeDepth, (2, 2), strides=(2, 2), activation='relu', padding='same'))
            convolutionalTransposeLayers.append(Deconv2D(convolutionalTransposeDepth, (3, 3), activation='relu', padding='same'))
            if i < self.__numberConvolutions - 1:
                convolutionalTransposeDepth //= 2
            else:
                convolutionalTransposeDepth = 1
            convolutionalTransposeLayers.append(Deconv2D(convolutionalTransposeDepth, (3, 3), activation='relu', padding='same'))


        def convolutionalTransposesAndUpSampling(predeconvolutionalRepresentation):
            return self.evaluateLayersList(convolutionalTransposeLayers, predeconvolutionalRepresentation)

        return convolutionalTransposesAndUpSampling
