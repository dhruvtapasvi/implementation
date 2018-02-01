import numpy as np
from keras.metrics import binary_crossentropy
from keras.backend import mean, flatten


def reconstructionLossConstructor(inputRepresentationDimensions):
    totalNumberOfPixels = np.prod(inputRepresentationDimensions)

    def reconstructionLoss(inputRepresentation, decodedInputRepresentation):
        return mean(totalNumberOfPixels * binary_crossentropy(flatten(inputRepresentation), flatten(decodedInputRepresentation)))

    return reconstructionLoss
