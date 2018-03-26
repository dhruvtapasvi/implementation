from keras.backend import sum, square, mean, log


def kullbackLeiberLossConstructor(latentRepresentationMean, latentRepresentationVarianceInput):
    epsilon = 1e-5
    latentRepresentationVariance = latentRepresentationVarianceInput + epsilon

    def kullbackLeiberLoss(inputRepresentation, decodedInputRepresentation):
        return mean(- 0.5 * sum(1 + log(latentRepresentationVariance) - square(latentRepresentationMean) - latentRepresentationVariance, axis=-1))

    return kullbackLeiberLoss
