from model.VariationalAutoencoder import VariationalAutoencoder
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from model.sampling import sampleConstructor
from model.VariationalAutoencoderLoss import VariationalAutoencoderLoss
from model.AlreadyTrainedError import AlreadyTrainedError
import numpy as np


class MnistDenseAutoencoder(VariationalAutoencoder):
    # Architecture from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    def __init__(self, originalImageDimensions, intermediateDimension, latentDimension):
        flattenedImageDimension = originalImageDimensions[0] * originalImageDimensions[1]

        inputImage = Input(shape=originalImageDimensions)
        flattenedInput = Flatten()(inputImage)
        intermediate = Dense(intermediateDimension, activation='relu')(flattenedInput)
        latentMean = Dense(latentDimension, activation='relu')(intermediate)
        latentLogVariance = Dense(latentDimension, activation='relu')(intermediate)

        self._encoder = Model(inputImage, latentMean)

        latent = Lambda(sampleConstructor(latentDimension), output_shape=(latentDimension,))([latentMean, latentLogVariance])
        intermediateDecoder = Dense(intermediateDimension, activation='relu')
        flattenedInputDecoder = Dense(flattenedImageDimension, activation='relu')
        flattenedInputReshaper = Reshape(originalImageDimensions)

        decodedIntermediate = intermediateDecoder(latent)
        decodedFlattenedInput = flattenedInputDecoder(decodedIntermediate)
        decodedInput = flattenedInputReshaper(decodedFlattenedInput)
        lossLayer = VariationalAutoencoderLoss(flattenedImageDimension, latentMean, latentLogVariance)([inputImage, decodedInput])
        self._autoencoder = Model(inputImage, lossLayer)

        customLatent = Input(shape=(latentDimension,))
        decodedCustomIntermediate = intermediateDecoder(customLatent)
        decodedFlattenedCustomInput = flattenedInputDecoder(decodedCustomIntermediate)
        decodedCustomInput = flattenedInputReshaper(decodedFlattenedCustomInput)
        self._decoder = Model(customLatent, decodedCustomInput)

        self._isTrained = False

    def encoder(self) -> Model:
        return self._encoder

    def decoder(self) -> Model:
        return self._autoencoder

    def autoencoder(self) -> Model:
        return self._autoencoder

    def train(
            self,
            trainingData: np.ndarray,
            validationData: np.ndarray,
            epochs,
            batchSize):
        if self._isTrained:
            raise AlreadyTrainedError
        else:
            self._autoencoder.compile(optimizer='rmsprop', loss=None)
            self._autoencoder.fit(
                trainingData,
                shuffle=True,
                epochs=epochs,
                batch_size=batchSize,
                validation_data=(validationData, None))

    def summary(self):
        self._autoencoder.summary()
