from model.VariationalAutoencoder import VariationalAutoencoder
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from model.sampling import samplingConstructor
from model.variationalAutoencoderLoss import variationalAutoencoderLoss
from model.AlreadyTrainedError import AlreadyTrainedError
import numpy as np

class MnistDenseAutoencoder(VariationalAutoencoder):
    # Architecture from https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
    def __init__(self, inputRepresentationDimensions, intermediateRepresentationDimension, latentRepresentationDimension):
        totalNumberOfPixels = inputRepresentationDimensions[0] * inputRepresentationDimensions[1]

        inputRepresentation = Input(shape=inputRepresentationDimensions)
        flattenedInputRepresentation = Flatten()(inputRepresentation)
        intermediateRepresentation = Dense(intermediateRepresentationDimension, activation='relu')(flattenedInputRepresentation)
        latentRepresentationMean = Dense(latentRepresentationDimension, activation='relu')(intermediateRepresentation)
        latentRepresentationLogVariance = Dense(latentRepresentationDimension, activation='relu')(intermediateRepresentation)

        self._encoder = Model(inputRepresentation, latentRepresentationMean)

        latentRepresentation = Lambda(samplingConstructor(latentRepresentationDimension), output_shape=(latentRepresentationDimension,))([latentRepresentationMean, latentRepresentationLogVariance])
        latentToIntermediate = Dense(intermediateRepresentationDimension, activation='relu')
        intermediateToFlattenedInput = Dense(totalNumberOfPixels, activation='sigmoid')
        flattenedInputToInput = Reshape(inputRepresentationDimensions)

        decodedIntermediateRepresentation = latentToIntermediate(latentRepresentation)
        decodedFlattenedInputRepresentation = intermediateToFlattenedInput(decodedIntermediateRepresentation)
        decodedInputRepresentation = flattenedInputToInput(decodedFlattenedInputRepresentation)
        self._autoencoder = Model(inputRepresentation, decodedInputRepresentation)

        self._autoencoder.add_loss(variationalAutoencoderLoss(
            inputRepresentationDimensions,
            inputRepresentation,
            decodedInputRepresentation,
            latentRepresentationMean,
            latentRepresentationLogVariance
        ))
        self._autoencoder.compile(optimizer='rmsprop', loss=None)

        customLatent = Input(shape=(latentRepresentationDimension,))
        decodedCustomIntermediate = latentToIntermediate(customLatent)
        decodedFlattenedCustomInput = intermediateToFlattenedInput(decodedCustomIntermediate)
        decodedCustomInput = flattenedInputToInput(decodedFlattenedCustomInput)
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
