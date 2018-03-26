import numpy as np
from keras.layers import Dense, Flatten, Reshape, Conv2D, Deconv2D, BatchNormalization, Dropout, Concatenate

from model.VariationalAutoencoder import VariationalAutoencoder


class ConvolutionalAutoencoderUnetConnections(VariationalAutoencoder):
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
        self.__convolutionResults = {}

    def buildModels(self):
        autoencoderEncoderLayers, encoderEncoderLayers = self.encoderLayersConstructor()
        autoencoderDecoderLayers, decoderDecoderLayers = self.decoderLayersConstructor()

        self.buildAutoencoder(autoencoderEncoderLayers, autoencoderDecoderLayers)
        self.buildEncoder(encoderEncoderLayers)
        self.buildDecoder(decoderDecoderLayers)

    def encoderLayersConstructor(self):
        convolutionalShape = self.__inputRepresentationDimensions + (1,)
        preContractionLayersList = [
            Reshape(convolutionalShape),
            BatchNormalization()
        ]
        contractionPhaseLayersConstructor = self.__buildContractionPhaseLayerConstructor()
        autoencoderContractionPhase = [contractionPhaseLayersConstructor(True)]
        encoderContractionPhase = [contractionPhaseLayersConstructor(False)]
        postContractionLayersList = [
            Flatten(),
            Dense(self.__intermediateRepresentationDimension, activation='relu', kernel_initializer="he_normal", bias_initializer="uniform"),
            BatchNormalization(),
            Dropout(self.__dropout)
        ]
        autoencoderEncoderLayers = preContractionLayersList + autoencoderContractionPhase + postContractionLayersList
        encoderEncoderLayers = preContractionLayersList + encoderContractionPhase + postContractionLayersList
        intermediateToLatentMean = Dense(self.__latentRepresentationDimension)
        intermediateToLatentLogVariance = Dense(self.__latentRepresentationDimension)

        def fullEncoderLayersConstructor(preintermediateLayers):
            def fullEncoderLayers(inputRepresentation):
                intermediateRepresentation = self.evaluateLayersList(preintermediateLayers, inputRepresentation)
                latentRepresentationMean = intermediateToLatentMean(intermediateRepresentation)
                latentRepresentationLogVariance = intermediateToLatentLogVariance(intermediateRepresentation)
                return latentRepresentationMean, latentRepresentationLogVariance

            return fullEncoderLayers

        return fullEncoderLayersConstructor(autoencoderEncoderLayers), fullEncoderLayersConstructor(encoderEncoderLayers)

    def decoderLayersConstructor(self):
        shrinkFactor = 2 ** (self.__numberConvolutions - 1)
        convolutionalTransposeDimensions = tuple(map(lambda x: x // shrinkFactor, self.__inputRepresentationDimensions)) + (shrinkFactor * self.__baseConvolutionalDepth,)
        totalNumberOfNodes = np.prod(convolutionalTransposeDimensions)
        preExpansionLayersList = [
            Dense(self.__intermediateRepresentationDimension, activation='relu', kernel_initializer="he_normal", bias_initializer="uniform"),
            BatchNormalization(),
            Dropout(self.__dropout),
            Dense(totalNumberOfNodes, activation='relu', kernel_initializer="he_normal", bias_initializer="uniform"),
            BatchNormalization(),
            Dropout(self.__dropout),
            Reshape(convolutionalTransposeDimensions)
        ]
        expansionPhaseLayerConstructor = self.__buildExpansionPhaseLayerConstructor()
        autoencoderExpansionPhaseLayers = [expansionPhaseLayerConstructor(True)]
        decoderExapansionPhaseLayers = [expansionPhaseLayerConstructor(False)]
        postExpansionLayersList = [
            Deconv2D(1, (3, 3), padding="same", activation="sigmoid", kernel_initializer="glorot_normal", bias_initializer="uniform"),
            Reshape(self.__inputRepresentationDimensions)
        ]
        autoencoderDecoderLayers = preExpansionLayersList + autoencoderExpansionPhaseLayers + postExpansionLayersList
        decoderDecoderLayers = preExpansionLayersList + decoderExapansionPhaseLayers + postExpansionLayersList

        return self.combineLayersIntoSingleLayer(autoencoderDecoderLayers), self.combineLayersIntoSingleLayer(decoderDecoderLayers)

    def __buildContractionPhaseLayerConstructor(self):
        layersList = [
            self.__buildDownsamplingAndConvolutionalsLayer(
                self.__baseConvolutionalDepth * (2 ** i),
                downsample=(i != 0)
            ) for i in range(self.__numberConvolutions)
        ]

        def buildContractionPhaseLayer(unetConnections=False):
            if unetConnections:
                def contractionPhaseLayer(layerInput):
                    layerOutput = layerInput
                    for (index, layer) in enumerate(layersList):
                        layerOutput = layer(layerOutput)
                        self.__convolutionResults[index] = layerOutput
                    return layerOutput

                return contractionPhaseLayer
            else:
                return self.combineLayersIntoSingleLayer(layersList)

        return buildContractionPhaseLayer

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

    def __buildExpansionPhaseLayerConstructor(self):
        layersList = [
            self.__buildDeconvolutionsAndUpsamplingLayer(
                self.__baseConvolutionalDepth * (2 ** i),
                upsample=(i != 0)
            ) for i in range(self.__numberConvolutions - 1, -1, -1)
        ]

        def buildExpansionPhaseLayer(unetConnections=False):
            if unetConnections:
                def expansionPhaseLayer(layerInput):
                    layerOutput = layerInput
                    for index, layer in enumerate(layersList):
                        layerOutput = Concatenate(axis=-1)([self.__convolutionResults[self.__numberConvolutions - index - 1], layerOutput])

                        layerOutput = layer(layerOutput)
                    return layerOutput

                return expansionPhaseLayer
            else:
                return self.combineLayersIntoSingleLayer(layersList)

        return buildExpansionPhaseLayer

    def __buildDeconvolutionsAndUpsamplingLayer(self, filters, numDeconvAfterUpsamping=2, upsample=True):
        convolutions = [self.__buildDeepDeconvolutionalLayer(filters) for _ in range(numDeconvAfterUpsamping)]
        upsampling = [self.__buildDeepDeconvolutionalLayer(filters // 2, (2, 2), (2, 2))] if upsample else []
        layersList = convolutions + upsampling
        return self.combineLayersIntoSingleLayer(layersList)


    def __buildDeepDeconvolutionalLayer(self, filters, kernelSize=(3, 3), strides=(1, 1)):
        layersList = [
            Deconv2D(filters, kernelSize, input_shape=(None, None, None), strides=strides, activation="relu", padding="same", kernel_initializer="he_normal", bias_initializer="uniform"),
            BatchNormalization(),
            Dropout(self.__dropout)
        ]
        return self.combineLayersIntoSingleLayer(layersList)
