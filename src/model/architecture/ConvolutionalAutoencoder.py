import numpy as np
from keras.layers import Dense, Flatten, Reshape, Conv2D, Deconv2D, BatchNormalization, Dropout

from model.VariationalAutoencoder import VariationalAutoencoder


class ConvolutionalAutoencoder(VariationalAutoencoder):
    # Architecture from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    def __init__(
            self,
            reconstructionLossConstructor,
            klLossWeight,
            inputRepresentationDimensions,
            numberConvolutions,
            baseConvolutionalDepth,
            intermediateRepresentationDimension,
            latentRepresentationDimension,
            dropout=0.0):
        super().__init__(reconstructionLossConstructor, klLossWeight, inputRepresentationDimensions, latentRepresentationDimension)
        self.__baseConvolutionalDepth = baseConvolutionalDepth
        self.__numberConvolutions = numberConvolutions
        self.__inputRepresentationDimensions = inputRepresentationDimensions
        self.__intermediateRepresentationDimension = intermediateRepresentationDimension
        self.__latentRepresentationDimension = latentRepresentationDimension
        self.__dropout = dropout

    def encoderLayersConstructor(self):
        convolutionalShape = self.__inputRepresentationDimensions + (1,)
        encoderLayersList = [
            Reshape(convolutionalShape),
            BatchNormalization(),
            self.__buildContractionPhaseLayer(),
            Flatten(),
            Dense(self.__intermediateRepresentationDimension, activation='relu', kernel_initializer="he_normal", bias_initializer="uniform"),
            BatchNormalization(),
            Dropout(self.__dropout)
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
        totalNumberOfNodes = np.prod(convolutionalTransposeDimensions)

        decoderLayersList = [
            Dense(self.__intermediateRepresentationDimension, activation='relu', kernel_initializer="he_normal", bias_initializer="uniform"),
            BatchNormalization(),
            Dropout(self.__dropout),
            Dense(totalNumberOfNodes, activation='relu', kernel_initializer="he_normal", bias_initializer="uniform"),
            BatchNormalization(),
            Dropout(self.__dropout),
            Reshape(convolutionalTransposeDimensions),
            self.__buildExpansionPhaseLayer(),
            Deconv2D(1, (3, 3), padding="same", activation="sigmoid", kernel_initializer="glorot_normal", bias_initializer="uniform"),
            Reshape(self.__inputRepresentationDimensions)
        ]

        return self.combineLayersIntoSingleLayer(decoderLayersList)

    def __buildContractionPhaseLayer(self):
        layersList = [
            self.__buildDownsamplingAndConvolutionalsLayer(
                self.__baseConvolutionalDepth * (2 ** i),
                downsample=(i != 0)
            ) for i in range(self.__numberConvolutions)
        ]
        return self.combineLayersIntoSingleLayer(layersList)

    def __buildDownsamplingAndConvolutionalsLayer(self, filters, numConvAfterDownsampling=2, downsample=True):
        downsampling = [self.__buildDeepConvolutionalLayer(filters // 2, (2, 2), (2, 2))] if downsample else []
        convolutions = [self.__buildDeepConvolutionalLayer(filters) for _ in range(numConvAfterDownsampling)]
        layersList = downsampling + convolutions
        return self.combineLayersIntoSingleLayer(layersList)


    def __buildDeepConvolutionalLayer(self, filters, kernelSize=(3, 3), strides=(1, 1)):
        layersList = [
            Conv2D(filters, kernelSize, strides=strides, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="uniform"),
            BatchNormalization(),
            Dropout(self.__dropout)
        ]
        return self.combineLayersIntoSingleLayer(layersList)

    def __buildExpansionPhaseLayer(self):
        layersList = [
            self.__buildDeconvolutionsAndUpsamplingLayer(
                self.__baseConvolutionalDepth * (2 ** i),
                upsample=(i != self.__numberConvolutions - 1)
            ) for i in range(self.__numberConvolutions - 1, -1, -1)
        ]
        return self.combineLayersIntoSingleLayer(layersList)


    def __buildDeconvolutionsAndUpsamplingLayer(self, filters, numDeconvAfterUpsamping=2, upsample=True):
        upsampling = [self.__buildDeepDeconvolutionalLayer(filters, (2, 2), (2, 2))] if upsample else []
        convolutions = [self.__buildDeepDeconvolutionalLayer(filters) for _ in range(numDeconvAfterUpsamping)]
        layersList = upsampling + convolutions
        return self.combineLayersIntoSingleLayer(layersList)


    def __buildDeepDeconvolutionalLayer(self, filters, kernelSize=(3, 3), strides=(1, 1)):
        layersList =[
            Deconv2D(filters, kernelSize, strides=strides, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="uniform"),
            BatchNormalization(),
            Dropout(self.__dropout)
        ]
        return self.combineLayersIntoSingleLayer(layersList)
