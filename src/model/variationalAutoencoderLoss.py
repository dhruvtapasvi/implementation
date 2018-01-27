from keras.metrics import binary_crossentropy
from keras.backend import sum, exp, mean, square, flatten


def variationalAutoencoderLoss(inputRepresentationDimensions, inputRepresentation, decodedInputRepresentation, latentRepresentationMean, latentRepresentationLogVariance):
    totalNumberOfPixels = inputRepresentationDimensions[0] * inputRepresentationDimensions[1]
    crossEntropyLoss = totalNumberOfPixels * binary_crossentropy(flatten(inputRepresentation), flatten(decodedInputRepresentation))
    kullbackLeiberLoss = - 0.5 * sum(1 + latentRepresentationLogVariance - square(latentRepresentationMean) - exp(latentRepresentationLogVariance), axis=-1)
    return mean(crossEntropyLoss + kullbackLeiberLoss)
