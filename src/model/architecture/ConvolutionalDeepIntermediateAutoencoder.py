import numpy as np
from keras.layers import Dense, Flatten, Reshape, Conv2D, Deconv2D, BatchNormalization

from model.VariationalAutoencoder import VariationalAutoencoder


class ConvolutionalDeepIntermediateAutoencoder(VariationalAutoencoder):
    def __init__(
            self,
            reconstructionLossConstructor,
            klLossWeight,
            inputDimensions,
            numberConvolutions,
            downsampleLast,
            baseConvolutionalDepth,
            encoderDenseDimensions,
            decoderDenseDimensions,
            latentDimension):
        super().__init__(reconstructionLossConstructor, klLossWeight, inputDimensions, latentDimension)
        self.__baseConvolutionalDepth = baseConvolutionalDepth
        self.__numberConvolutions = numberConvolutions
        self.__downsampleLast = downsampleLast
        self.__inputDimensions = inputDimensions
        self.__encoderDenseDimensions = encoderDenseDimensions
        self.__decoderDenseDimensions = decoderDenseDimensions
        self.__latentDimension = latentDimension

    def encoderLayersConstructor(self):
        convolutionalShape = self.__inputDimensions + (1,)
        encoderLayersList = [
            Reshape(convolutionalShape),
            BatchNormalization(),
            self.__encoderConvolutionsConstructor(),
            Flatten(),
            self.__denseLayersConstructor(self.__encoderDenseDimensions),
        ]
        intermediateToLatentMean = Dense(self.__latentDimension)
        intermediateToLatentLogVariance = Dense(self.__latentDimension)

        def encoderLayers(inputRepresentation):
            intermediate = self.evaluateLayersList(encoderLayersList, inputRepresentation)
            latentMean = intermediateToLatentMean(intermediate)
            latentLogVariance = intermediateToLatentLogVariance(intermediate)
            return latentMean, latentLogVariance

        return encoderLayers

    def decoderLayersConstructor(self):
        shrinkFactor = 2 ** (self.__numberConvolutions - 1)
        maxNumFiltersConv = shrinkFactor * self.__baseConvolutionalDepth
        minImageSizeConv = tuple(map(lambda x: x // shrinkFactor, self.__inputDimensions))
        maxNumFilters = maxNumFiltersConv * (2 if self.__downsampleLast else 1)
        minImageSize = tuple(map(lambda x: x // (2 if self.__downsampleLast else 1), minImageSizeConv))
        convolutionalTransposeDimensions = minImageSize + (maxNumFilters,)
        totalNumberOfNodes = np.prod(convolutionalTransposeDimensions)
        decoderLayersList = [
            BatchNormalization(),
            self.__denseLayersConstructor(self.__decoderDenseDimensions),
            Dense(totalNumberOfNodes, activation='relu', kernel_initializer="he_normal"),
            BatchNormalization(),
            Reshape(convolutionalTransposeDimensions),
            self.__decoderDeconvolutionsConstructor(minImageSizeConv, maxNumFiltersConv),
            Deconv2D(1, (3, 3), padding="same", activation="sigmoid", kernel_initializer="glorot_normal"),
            Reshape(self.__inputDimensions)
        ]

        return self.collapseLayers(decoderLayersList)

    def __encoderConvolutionsConstructor(self):
        numFilters = self.__baseConvolutionalDepth
        imageDimensions = self.__inputDimensions
        layers = []
        for i in range(self.__numberConvolutions):
            kernelSize = tuple(map(min, zip(imageDimensions, (3, 3))))
            layers.append(self.__convolutionalModuleConstructor(
                numFilters,
                kernelSize,
                i < (self.__numberConvolutions - 1) or self.__downsampleLast
            ))
            numFilters *= 2
            imageDimensions = tuple(map(lambda x: x // 2, imageDimensions))
        return self.collapseLayers(layers)

    def __decoderDeconvolutionsConstructor(self, initialImageDimensions, initialNumFilters):
        layers = []
        for i in range(self.__numberConvolutions):
            kernelSize = tuple(map(min, zip(initialImageDimensions, (3, 3))))
            layers.append(self.__deconvolutionalModuleConstructor(
                initialNumFilters,
                kernelSize,
                i > 0 or self.__downsampleLast
            ))
            initialNumFilters //= 2
            initialImageDimensions = tuple(map(lambda x: x * 2, initialImageDimensions))
        return self.collapseLayers(layers)

    def __denseLayersConstructor(self, sizes):
        return self.collapseLayers([self.__denseLayerConstructor(size) for size in sizes])

    def __convolutionalModuleConstructor(self, filters, nonDownSamplingKernelSize=(3, 3), downSample=True):
        layers = [
            self.__convolutionalLayerConstructor(filters, nonDownSamplingKernelSize, (1, 1), 'same'),
            self.__convolutionalLayerConstructor(filters, nonDownSamplingKernelSize, (1, 1), 'same')
        ] + ([
            self.__convolutionalLayerConstructor(filters * 2, (2, 2), (2, 2), 'valid')
        ] if downSample else [])
        return self.collapseLayers(layers)

    def __deconvolutionalModuleConstructor(self, filters, nonUpSamplingKernelSize, upSample=True):
        layers = ([
            self.__deconvolutionalLayerConstructor(filters, (2, 2), (2, 2), 'valid')
        ] if upSample else []) + [
            self.__deconvolutionalLayerConstructor(filters, nonUpSamplingKernelSize, (1, 1), 'same'),
            self.__deconvolutionalLayerConstructor(filters, nonUpSamplingKernelSize, (1, 1), 'same')
        ]
        return self.collapseLayers(layers)

    def __denseLayerConstructor(self, size):
        layers = [
            Dense(size, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization()
        ]
        return self.collapseLayers(layers)

    def __convolutionalLayerConstructor(self, filters, kernelSize, strides, padding):
        layers = [
            Conv2D(filters=filters, kernel_size=kernelSize, strides=strides, padding=padding, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization()
        ]
        return self.collapseLayers(layers)

    def __deconvolutionalLayerConstructor(self, filters, kernelSize, strides, padding):
        layers = [
            Deconv2D(filters=filters, kernel_size=kernelSize, strides=strides, padding=padding, activation='relu', kernel_initializer='he_normal'),
            BatchNormalization()
        ]
        return self.collapseLayers(layers)
