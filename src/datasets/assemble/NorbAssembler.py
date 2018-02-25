import numpy as np


class NorbAssembler:
    def _addLabel(self, label):
        return np.array([np.append([0], label), np.append([1], label)])

    def assemble(self, images, labels):
        xDim, yDim = images[0][0].shape
        numImages = images.shape[0] * 2
        newImages = images.reshape(numImages, xDim, yDim)

        numLabelDimensions = labels[0].shape[0]
        stereoLabelled = np.apply_along_axis(self._addLabel, 1, labels)
        newLabels = stereoLabelled.flatten().reshape(numImages, numLabelDimensions + 1)
        return newImages, newLabels
