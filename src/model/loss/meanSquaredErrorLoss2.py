import math
from keras.backend import mean, flatten, log, square, sum


def meanSquaredErrorLossConstructor(
        decodedInputRepresentationVariance=None,
        outputVarianceFixed=1.0,
        minimumVarianceEpsilon=10e-3):

    def meanSquaredErrorLoss(inputRepresentation, decodedInputRepresentation):
        flattenedInputRepresentation = flatten(inputRepresentation)
        flattenedDecodedInputRepresentation = flatten(decodedInputRepresentation)

        constantLossPerPixel = 0.5 * math.log(2 * math.pi)
        varianceLossPerPixel = (
            0.5 * math.log(outputVarianceFixed + minimumVarianceEpsilon)
        ) if decodedInputRepresentationVariance is None else (
            0.5 * log(flatten(decodedInputRepresentationVariance) + minimumVarianceEpsilon)
        )
        reconstructionLossPerPixel = (
            0.5 * square(flattenedDecodedInputRepresentation - flattenedInputRepresentation) / (outputVarianceFixed + minimumVarianceEpsilon)
        ) if decodedInputRepresentationVariance is None else (
            0.5 * square(flattenedDecodedInputRepresentation - flattenedInputRepresentation) / (flatten(decodedInputRepresentationVariance) + minimumVarianceEpsilon)
        )
        totalLossPerPixel = constantLossPerPixel + varianceLossPerPixel + reconstructionLossPerPixel
        imageLoss = sum(totalLossPerPixel, axis=-1)
        return mean(imageLoss)

    return meanSquaredErrorLoss
