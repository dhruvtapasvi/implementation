from model.loss.kullbackLeiberLoss import kullbackLeiberLossConstructor
from model.loss.binaryCrossEntropyLoss import binaryCrossEntropyLossConstructor


def variationalAutoencoderLossConstructor(inputRepresentationDimensions, latentRepresentationMean, latentRepresentationLogVariance):
    reconstructionLoss = binaryCrossEntropyLossConstructor(inputRepresentationDimensions)
    kullbackLeiberLoss = kullbackLeiberLossConstructor(latentRepresentationMean, latentRepresentationLogVariance)

    def variationalAutoencoderLoss(inputRepresentation, decodedInputRepresentation):
        return reconstructionLoss(inputRepresentation, decodedInputRepresentation) + kullbackLeiberLoss(inputRepresentation, decodedInputRepresentation)

    return variationalAutoencoderLoss
