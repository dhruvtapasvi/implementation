from model.loss.kullbackLeiberLoss import kullbackLeiberLossConstructor


def variationalAutoencoderLossConstructor(
        reconstructionLossConstructor,
        klLossWeight,
        inputRepresentationDimensions,
        latentRepresentationMean,
        latentRepresentationLogVariance):
    reconstructionLoss = reconstructionLossConstructor(inputRepresentationDimensions)
    kullbackLeiberLoss = kullbackLeiberLossConstructor(latentRepresentationMean, latentRepresentationLogVariance)

    def variationalAutoencoderLoss(inputRepresentation, decodedInputRepresentation):
        return reconstructionLoss(inputRepresentation, decodedInputRepresentation) + klLossWeight * kullbackLeiberLoss(inputRepresentation, decodedInputRepresentation)

    return variationalAutoencoderLoss
