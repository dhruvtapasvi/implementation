import math
import random

import numpy as np
from skimage.transform import AffineTransform, warp
from skimage.util import pad, img_as_ubyte

from dataset.loader.DatasetLoader import DatasetLoader


class RandomTransforms(DatasetLoader):
    def __init__(self, baseDatasetLoader: DatasetLoader, shearFactor, maxAbsStretchExponent, numSamplesPerTrainPoint, numSamplesPerValidationPoint=None, numSamplesPerTestPoint=None, randomSeed=None):
        self.__baseDatasetLoader = baseDatasetLoader
        self.__shearRange = shearFactor
        self.__stretchExponentRange = maxAbsStretchExponent
        self.__numSamplesPerTrainPoint = numSamplesPerTrainPoint
        self.__numSamplesPerValidationPoint = numSamplesPerTrainPoint if numSamplesPerValidationPoint is None else numSamplesPerValidationPoint
        self.__numSamplesPerTestPoint = numSamplesPerTrainPoint if numSamplesPerTestPoint is None else numSamplesPerTestPoint
        self.__imageGrowFactor = (1.0 + shearFactor) * (2 ** (0.5 + maxAbsStretchExponent))
        self.__randomSeed = randomSeed

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        random.seed(self.__randomSeed)
        (xTrain, yTrain), (xVal, yVal), (xTest, yTest) = self.__baseDatasetLoader.loadData()
        train = self.__randomTransform(xTrain, yTrain, self.__numSamplesPerTrainPoint)
        validation = self.__randomTransform(xVal, yVal, self.__numSamplesPerValidationPoint)
        test = self.__randomTransform(xTest, yTest, self.__numSamplesPerTestPoint)
        return train, validation, test

    def __randomTransform(self, X: np.ndarray, Y: np.ndarray, numSamplesPerPoint):
        newImageDimensions = self.dataPointShape()[0][0:2]
        oldImageDimensions = self.__baseDatasetLoader.dataPointShape()[0][0:2]

        firstAxisPadding = (newImageDimensions[0] - oldImageDimensions[0]) // 2
        secondAxisPadding = (newImageDimensions[1] - oldImageDimensions[1]) // 2
        paddingDimensions = ((firstAxisPadding, firstAxisPadding), (secondAxisPadding, secondAxisPadding))

        translationVector = tuple(map(lambda x: x / 2.0 - 0.5, newImageDimensions))
        minusTranslationVector = tuple(map(lambda x: -x, translationVector))
        translateToCentre = AffineTransform(translation=minusTranslationVector)
        translationBack = AffineTransform(translation=translationVector)

        XTransformed = []
        YTransformed = []

        for index, x in enumerate(X):
            xPadded = pad(x, paddingDimensions, 'constant')
            y = tuple(Y[index])
            for _ in range(numSamplesPerPoint):
                randomRotationAngle = random.uniform(0.0, 2.0 * math.pi)
                randomShearFactor = random.uniform(-self.__shearRange, self.__shearRange)
                randomStretchExponentFirstAxis = random.uniform(-self.__stretchExponentRange, self.__stretchExponentRange)
                randomStretchFactorFirstAxis = 2.0 ** randomStretchExponentFirstAxis
                randomStretchExponentSecondAxis = random.uniform(-self.__stretchExponentRange, self.__stretchExponentRange)
                randomStretchFactorSecondAxis = 2.0 ** randomStretchExponentSecondAxis

                randomTransformation = AffineTransform(
                    rotation=randomRotationAngle,
                    shear=randomShearFactor,
                    scale=(randomStretchFactorFirstAxis, randomStretchFactorSecondAxis)
                )
                combinedInverseTransform = (translateToCentre + (randomTransformation + translationBack)).inverse
                xTransformed = warp(xPadded, combinedInverseTransform)
                XTransformed.append(img_as_ubyte(xTransformed))

                YTransformed.append((index,) + y + (randomRotationAngle, randomShearFactor, randomStretchExponentFirstAxis, randomStretchExponentSecondAxis))

        return np.array(XTransformed), np.array(YTransformed)

    def dataPointShape(self):
        maxDimension = max(self.__baseDatasetLoader.dataPointShape()[0][0:2])
        newMaxDimension = math.ceil(self.__imageGrowFactor * maxDimension)
        newImageDimensions = (newMaxDimension, newMaxDimension) + self.__baseDatasetLoader.dataPointShape()[0][2:]
        newLabelDimensions = (self.__baseDatasetLoader.dataPointShape()[1][0] + 5,)
        return newImageDimensions, newLabelDimensions
