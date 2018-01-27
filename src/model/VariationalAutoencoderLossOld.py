from keras.layers import Layer
import keras.backend as K
from keras.metrics import binary_crossentropy


class VariationalAutoencoderLossOld(Layer):
    def __init__(self, originalDimension, latentMean, latentVariance, **kwargs):
        self.is_placeholder = True
        self._originalDimension = originalDimension
        self._latentMean = latentMean
        self._latentVariance = latentVariance
        super(VariationalAutoencoderLossOld, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = self._originalDimension * binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self._latentVariance - K.square(self._latentMean) - K.exp(self._latentVariance), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x
