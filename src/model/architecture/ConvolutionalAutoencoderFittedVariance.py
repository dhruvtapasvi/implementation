import numpy as np
from keras.layers import Dense, Flatten, Reshape, Conv2D, Deconv2D, BatchNormalization

from model.VariationalAutoencoderFittedVariance import VariationalAutoencoderFittedVariance


class ConvolutionalAutoencoderFittedVariance(VariationalAutoencoderFittedVariance):
    # Architecture from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    def __init__(self, reconstructionLossConstructor, klLossWeight, inputRepresentationDimensions, numberConvolutions, baseConvolutionalDepth, intermediateRepresentationDimension, latentRepresentationDimension):
        super().__init__(reconstructionLossConstructor, klLossWeight, inputRepresentationDimensions, latentRepresentationDimension)
        self.__baseConvolutionalDepth = baseConvolutionalDepth
        self.__numberConvolutions = numberConvolutions
        self.__inputRepresentationDimensions = inputRepresentationDimensions
        self.__intermediateRepresentationDimension = intermediateRepresentationDimension
        self.__latentRepresentationDimension = latentRepresentationDimension

    def encoderLayersConstructor(self):
        convolutionalShape = self.__inputRepresentationDimensions + (1,)
        encoderLayersList = [
            Reshape(convolutionalShape),
            BatchNormalization(),
            self.__buildConvolutionalsAndDownSamplings(),
            Flatten(),
            Dense(self.__intermediateRepresentationDimension, activation='relu'),
            BatchNormalization()
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
            BatchNormalization(),
            Dense(totalNumberOfNodes, activation='relu'),
            BatchNormalization(),
            Reshape(convolutionalTransposeDimensions),
            self.__buildConvolutionalTransposesAndUpsampling(),
        ]
        decodedInputMeanLayer = Deconv2D(1, (3,3), activation='sigmoid', padding='same', kernel_initializer='he_normal', bias_initializer='uniform')
        decodedInputMeanReshaper = Reshape(self.__inputRepresentationDimensions)
        decodedInputVarianceLayer = Deconv2D(1, (3,3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='uniform')
        decodedInputVarianceReshaper = Reshape(self.__inputRepresentationDimensions)


        def decoderLayers(latentRepresentation):
            intermediateRepresentation = self.evaluateLayersList(decoderLayersList, latentRepresentation)
            decodedInputMean = decodedInputMeanReshaper(decodedInputMeanLayer(intermediateRepresentation))
            decodedInputVariance = decodedInputVarianceReshaper(decodedInputVarianceLayer(intermediateRepresentation))
            return decodedInputMean, decodedInputVariance

        return decoderLayers

    def __buildConvolutionalsAndDownSamplings(self):
        convolutionalDepth = self.__baseConvolutionalDepth
        convolutionalLayers = []
        for i in range(self.__numberConvolutions):
            for j in range(2):
                convolutionalLayers.append(Conv2D(convolutionalDepth, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='uniform'))
                convolutionalLayers.append(BatchNormalization())
            if i < self.__numberConvolutions - 1:
                convolutionalLayers.append(Conv2D(convolutionalDepth, (2, 2), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='uniform'))
                convolutionalLayers.append(BatchNormalization())
                convolutionalDepth *= 2

        def convolutionsAndDownsampling(preconvolutionalRepresentation):
            return self.evaluateLayersList(convolutionalLayers, preconvolutionalRepresentation)

        return convolutionsAndDownsampling

    def __buildConvolutionalTransposesAndUpsampling(self):
        convolutionalTransposeDepth = self.__baseConvolutionalDepth * (2 ** (self.__numberConvolutions - 1))
        convolutionalTransposeLayers = []
        for i in range(self.__numberConvolutions):
            if i > 0:
                convolutionalTransposeLayers.append(Deconv2D(convolutionalTransposeDepth, (2, 2), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='uniform'))
                convolutionalTransposeLayers.append(BatchNormalization())
            convolutionalTransposeLayers.append(Deconv2D(convolutionalTransposeDepth, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='uniform'))
            convolutionalTransposeLayers.append(BatchNormalization())
            if i < self.__numberConvolutions - 1:
                convolutionalTransposeDepth //= 2
            else:
                convolutionalTransposeDepth = 1
            convolutionalTransposeLayers.append(Deconv2D(convolutionalTransposeDepth, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', bias_initializer='uniform'))
            convolutionalTransposeLayers.append(BatchNormalization())
        convolutionalTransposeLayers.pop()
        convolutionalTransposeLayers.pop()

        def convolutionalTransposesAndUpSampling(predeconvolutionalRepresentation):
            return self.evaluateLayersList(convolutionalTransposeLayers, predeconvolutionalRepresentation)

        return convolutionalTransposesAndUpSampling
