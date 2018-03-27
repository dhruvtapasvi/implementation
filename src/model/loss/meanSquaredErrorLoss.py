import numpy as np
import math
from keras.metrics import mean_squared_error
from keras.backend import mean, flatten, log, square, sum


def meanSquaredErrorLossConstructor(inputRepresentationDimensions, decodedInputRepresentationVariance=None):
    totalNumberOfPixels = np.prod(inputRepresentationDimensions)
    epsilon = 0.001

    def meanSquaredErrorLoss(inputRepresentation, decodedInputRepresentation):
        if decodedInputRepresentationVariance is not None:
            lossPerPixel = 0.5 * math.log(math.pi) + 0.5 * log(decodedInputRepresentationVariance + epsilon) + 0.5 * square(decodedInputRepresentation - inputRepresentation) / (decodedInputRepresentationVariance + epsilon)
            # lossPerPixel = square(decodedInputRepresentation - inputRepresentation)
            imageLoss = sum(lossPerPixel, axis=-1)
            return mean(imageLoss)
        else:
            # Assume variance is 1
            return mean(totalNumberOfPixels * 0.5 * (
                math.log(math.pi) + mean_squared_error(flatten(inputRepresentation), flatten(decodedInputRepresentation))
            ))

    return meanSquaredErrorLoss
