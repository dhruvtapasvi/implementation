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
        self._inputRepresentationDimensions = inputRepresentationDimensions
        self._intermediateRepresentationDimension = intermediateRepresentationDimension
        self._latentRepresentationDimension = latentRepresentationDimension

        encoderLayers = self._encoderLayersConstructor()
        decoderLayers = self._decoderLayersConstructor()

        self._buildAutoencoder(encoderLayers, decoderLayers)
        self._buildEncoder(encoderLayers)
        self._buildDecoder(decoderLayers)

        self._isTrained = False

    def _buildAutoencoder(self, encoderLayers, decoderLayers):
        # Input to the encoder and autoencoder models:
        inputRepresentation = Input(shape=self._inputRepresentationDimensions)

        latentRepresentationMean, latentRepresentationLogVariance = encoderLayers(inputRepresentation)
        latentRepresentation = Lambda(
            samplingConstructor(self._latentRepresentationDimension),
            output_shape=(self._latentRepresentationDimension,)
        )([latentRepresentationMean, latentRepresentationLogVariance])

        decodedInputRepresentation = decoderLayers(latentRepresentation)

        self._autoencoder = Model(inputRepresentation, decodedInputRepresentation)

        self._autoencoder.add_loss(variationalAutoencoderLoss(
            self._inputRepresentationDimensions,
            inputRepresentation,
            decodedInputRepresentation,
            latentRepresentationMean,
            latentRepresentationLogVariance
        ))
        self._autoencoder.compile(optimizer='rmsprop', loss=None)

    def _buildEncoder(self, encoderLayers):
        inputRepresentation = Input(shape=self._inputRepresentationDimensions)
        latentRepresentationMean, _ = encoderLayers(inputRepresentation)
        self._encoder = Model(inputRepresentation, latentRepresentationMean)

    def _buildDecoder(self, decoderLayers):
        customLatentRepresentation = Input(shape=(self._latentRepresentationDimension,))
        customDecodedInputRepresentation = decoderLayers(customLatentRepresentation)
        self._decoder = Model(customLatentRepresentation, customDecodedInputRepresentation)

    def _encoderLayersConstructor(self):
        inputToFlattenedInput = Flatten()
        flattenedInputToIntermediate = Dense(self._intermediateRepresentationDimension, activation='relu')
        intermediateToLatentMean = Dense(self._latentRepresentationDimension, activation='relu')
        intermediateToLatentLogVariance = Dense(self._latentRepresentationDimension, activation='relu')

        def encoderLayers(inputRepresentation):
            flattenedInputRepresentation = inputToFlattenedInput(inputRepresentation)
            intermediateRepresentation = flattenedInputToIntermediate(flattenedInputRepresentation)
            latentRepresentationMean = intermediateToLatentMean(intermediateRepresentation)
            latentRepresentationLogVariance = intermediateToLatentLogVariance(intermediateRepresentation)
            return latentRepresentationMean, latentRepresentationLogVariance

        return encoderLayers

    def _decoderLayersConstructor(self):
        totalNumberOfPixels = self._inputRepresentationDimensions[0] * self._inputRepresentationDimensions[1]
        latentToIntermediate = Dense(self._intermediateRepresentationDimension, activation='relu')
        intermediateToFlattenedInput = Dense(totalNumberOfPixels, activation='sigmoid')
        flattenedInputToInput = Reshape(self._inputRepresentationDimensions)

        def decoderLayers(latentRepresentation):
            decodedIntermediateRepresentation = latentToIntermediate(latentRepresentation)
            decodedFlattenedInputRepresentation = intermediateToFlattenedInput(decodedIntermediateRepresentation)
            decodedInputRepresentation = flattenedInputToInput(decodedFlattenedInputRepresentation)
            return decodedInputRepresentation

        return decoderLayers

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
