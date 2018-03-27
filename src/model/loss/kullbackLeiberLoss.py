from keras.backend import sum, square, mean, exp


def kullbackLeiberLossConstructor(latentRepresentationMean, latentLogVariance):
    def kullbackLeiberLoss(inputRepresentation, decodedInputRepresentation):
        return mean(- 0.5 * sum(1 + latentLogVariance - square(latentRepresentationMean) - exp(latentLogVariance), axis=-1))

    return kullbackLeiberLoss
