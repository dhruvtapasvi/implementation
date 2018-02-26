from model.loss.kullbackLeiberLoss import kullbackLeiberLossConstructor


def variationalAutoencoderLossConstructor(reconstructionLossConstructor, inputRepresentationDimensions, latentRepresentationMean, latentRepresentationLogVariance):
    reconstructionLoss = reconstructionLossConstructor(inputRepresentationDimensions)
    kullbackLeiberLoss = kullbackLeiberLossConstructor(latentRepresentationMean, latentRepresentationLogVariance)

    def variationalAutoencoderLoss(inputRepresentation, decodedInputRepresentation):
        return reconstructionLoss(inputRepresentation, decodedInputRepresentation) + kullbackLeiberLoss(inputRepresentation, decodedInputRepresentation)

    return variationalAutoencoderLoss
