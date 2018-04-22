from interpolate.Interpolate import Interpolate
from model.Autoencoder import Autoencoder


class InterpolateLatentSpace(Interpolate):
    def __init__(self, autoencoder: Autoencoder):
        self.__encoder = autoencoder.encoder()
        self.__decoder = autoencoder.decoder()

    def interpolateAll(self, left, right, intervals):
        leftLatent = self.__encoder.predict_on_batch(left)
        rightLatent = self.__encoder.predict_on_batch(right)
        interpolated = super(InterpolateLatentSpace, self).interpolateAll(leftLatent,rightLatent, intervals)
        flattenedInterpolated = interpolated.reshape((-1,) + interpolated.shape[2:])
        flattedReconstructed = self.__decoder.predict_on_batch(flattenedInterpolated)
        reconstructed = flattedReconstructed.reshape(interpolated.shape[0:2] + left.shape[1:])
        return interpolated, reconstructed
