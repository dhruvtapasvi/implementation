from model.VariationalAutoencoder import VariationalAutoencoder
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from model.sampling import sampleConstructor
from model.variationalAutoencoderLossConstructor import variationalAutoencoderLossConstructor
from model.AlreadyTrainedError import AlreadyTrainedError
import numpy as np

from keras.metrics import binary_crossentropy
from keras.backend import sum, exp, mean, square, flatten, shape, random_normal

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

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = random_normal(shape=(shape(z_mean)[0], latentDimension), mean=0.,
                                      stddev=1.0)
            return z_mean + exp(z_log_var / 2) * epsilon

        # latent = Lambda(sampleConstructor(latentDimension), output_shape=(latentDimension,))([latentMean, latentLogVariance])
        latent = Lambda(sampling, output_shape=(latentDimension,))([latentMean, latentLogVariance])
        intermediateDecoder = Dense(intermediateDimension, activation='relu')
        flattenedInputDecoder = Dense(flattenedImageDimension, activation='sigmoid')
        flattenedInputReshaper = Reshape(originalImageDimensions)

        decodedIntermediate = intermediateDecoder(latent)
        decodedFlattenedInput = flattenedInputDecoder(decodedIntermediate)
        decodedInput = flattenedInputReshaper(decodedFlattenedInput)
        self._autoencoder = Model(inputImage, decodedInput)

        crossEntropyLoss = 28 * 28 * binary_crossentropy(flatten(inputImage), flatten(decodedInput))
        kullbackLeiberLoss = - 0.5 * sum(1 + latentLogVariance - square(latentMean) - exp(latentLogVariance), axis=-1)
        vae_loss = mean(crossEntropyLoss + kullbackLeiberLoss)

        # self._autoencoder.add_loss(variationalAutoencoderLossConstructor(
        #     originalImageDimensions,
        #     inputImage,
        #     decodedInput,
        #     latentMean,
        #     latentLogVariance
        # ))
        self._autoencoder.add_loss(vae_loss)
        self._autoencoder.compile(optimizer='rmsprop', loss=None)

        customLatent = Input(shape=(latentDimension,))
        decodedCustomIntermediate = intermediateDecoder(customLatent)
        decodedFlattenedCustomInput = flattenedInputDecoder(decodedCustomIntermediate)
        decodedCustomInput = flattenedInputReshaper(decodedFlattenedCustomInput)
        self._decoder = Model(customLatent, decodedCustomInput)

        self._isTrained = False

    def encoder(self) -> Model:
        return self._encoder

    def decoder(self) -> Model:
        return self._decoder

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
            self._autoencoder.fit(
                trainingData,
                shuffle=True,
                epochs=epochs,
                batch_size=batchSize,
                validation_data=(validationData, None))

    def summary(self):
        self._autoencoder.summary()
