import numpy as np
from keras.metrics import binary_crossentropy
from keras.backend import mean, flatten


def binaryCrossEntropyLossConstructor(inputRepresentationDimensions):
    totalNumberOfPixels = np.prod(inputRepresentationDimensions)

    def binaryCrossEntropyLoss(inputRepresentation, decodedInputRepresentation):
        return mean(totalNumberOfPixels * binary_crossentropy(flatten(inputRepresentation), flatten(decodedInputRepresentation)))

    return binaryCrossEntropyLoss
