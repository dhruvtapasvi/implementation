from keras.metrics import binary_crossentropy
from keras.backend import sum, exp, mean, square, flatten


def variationalAutoencoderLossConstructor(imageDimensions, input, reconstructedInput, latentMean, latentVariance):
    crossEntropyLoss = imageDimensions[0] * imageDimensions[1] * binary_crossentropy(flatten(input), flatten(reconstructedInput))
    kullbackLeiberLoss = 0.5 * sum(1 + latentVariance - square(latentMean) - exp(latentVariance), axis=-1)
    return mean(crossEntropyLoss + kullbackLeiberLoss)
