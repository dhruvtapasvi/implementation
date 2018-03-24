from model.loss.kullbackLeiberLoss import kullbackLeiberLossConstructor


def variationalAutoencoderLossConstructor(
        reconstructionLossConstructor,
        klLossWeight,
        inputRepresentationDimensions,
        latentRepresentationMean,
        latentRepresentationLogVariance,
        decodedInputRepresentationVariance=None):
    reconstructionLoss = reconstructionLossConstructor(inputRepresentationDimensions, decodedInputRepresentationVariance)
    kullbackLeiberLoss = kullbackLeiberLossConstructor(latentRepresentationMean, latentRepresentationLogVariance)

    def variationalAutoencoderLoss(inputRepresentation, decodedInputRepresentation):
        return reconstructionLoss(inputRepresentation, decodedInputRepresentation) + klLossWeight * kullbackLeiberLoss(inputRepresentation, decodedInputRepresentation)
        # return reconstructionLoss(inputRepresentation, decodedInputRepresentation)

    return variationalAutoencoderLoss
