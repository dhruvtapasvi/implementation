import numpy as np
from keras.metrics import mean_squared_error
from keras.backend import mean, flatten


def meanSquaredErrorLossConstructor(inputRepresentationDimensions):
    totalNumberOfPixels = np.prod(inputRepresentationDimensions)

    def meanSquaredErrorLoss(inputRepresentation, decodedInputRepresentation):
        return mean(totalNumberOfPixels * mean_squared_error(flatten(inputRepresentation), flatten(decodedInputRepresentation)))

    return meanSquaredErrorLoss
