from keras.backend import sum, exp, square, mean


def kullbackLeiberLossConstructor(latentRepresentationMean, latentRepresentationLogVariance):
    def kullbackLeiberLoss(inputRepresentation, decodedInputRepresentation):
        return mean(- 0.5 * sum(1 + latentRepresentationLogVariance - square(latentRepresentationMean) - exp(latentRepresentationLogVariance), axis=-1))

    return kullbackLeiberLoss
