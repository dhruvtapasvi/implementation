from model.reconstructionLoss import reconstructionLossConstructor
from model.kullbackLeiberLoss import kullbackLeiberLossConstructor


def variationalAutoencoderLossConstructor(inputRepresentationDimensions, latentRepresentationMean, latentRepresentationLogVariance):
    reconstructionLoss = reconstructionLossConstructor(inputRepresentationDimensions)
    kullbackLeiberLoss = kullbackLeiberLossConstructor(latentRepresentationMean, latentRepresentationLogVariance)

    def variationalAutoencoderLoss(inputRepresentation, decodedInputRepresentation):
        return reconstructionLoss(inputRepresentation, decodedInputRepresentation) + kullbackLeiberLoss(inputRepresentation, decodedInputRepresentation)

    return variationalAutoencoderLoss
