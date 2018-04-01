import numpy as np
from skimage.transform import AffineTransform, warp
from skimage.util import pad, img_as_ubyte

from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.loader.DatasetLoader import DatasetLoader


class CreateTransformedInterpolateData(InterpolateDatasetLoader):
    def __init__(self, baseDatasetLoader: DatasetLoader, padding, rotationFactors, shearFactors, log2StretchFactors):
        #  Factors have (left, right, centre, outside) layout
        self.__baseDatasetLoader = baseDatasetLoader
        self.__padding = padding
        self.__rotationFactors = rotationFactors
        self.__defaultRotationFactor = 0.0
        self.__shearFactors = shearFactors
        self.__defaultShearFactor = 0.0
        self.__log2StretchFactors = log2StretchFactors
        self.__defaultLog2StretchFactor = 0.0

    def loadInterpolationData(self) -> (np.ndarray, np.ndarray):
        _, _, (xTest, yTest) = self.__baseDatasetLoader.loadData()
        xTestPadded = self.__pad(xTest)
        yTestLabelled = self.__label(yTest)
        xTestRotated, yTestRotated = self.__rotate(xTestPadded, yTestLabelled)
        xTestSheared, yTestSheared = self.__shear(xTestPadded, yTestLabelled)
        xTestStretched, yTestStretched = self.__stretch(xTestPadded, yTestLabelled)
        print("Concatenate")
        xTestTransformed = np.concatenate((xTestRotated, xTestSheared, xTestStretched))
        yTestTransformed = np.concatenate((yTestRotated, yTestSheared, yTestStretched))
        print("Done transforming!")
        return xTestTransformed, yTestTransformed

    def __pad(self, X: np.ndarray):
        print("Pad")
        XPadded = [pad(x, self.__padding, 'constant') for x in X]
        return np.array(XPadded)

    def __label(self, Y: np.ndarray):
        YLabelled = [
            (index,) + tuple(y) + (self.__defaultRotationFactor, self.__defaultShearFactor, self.__defaultLog2StretchFactor, self.__defaultLog2StretchFactor)
            for index, y in enumerate(Y)
        ]
        return np.array(YLabelled)

    def __rotate(self, X, Y):
        print("Rotate")
        return self.__transform(X, Y, self.__rotationFactors, lambda f: AffineTransform(rotation=f), (1,))

    def __shear(self, X, Y):
        print("Shear")
        return self.__transform(X, Y, self.__shearFactors, lambda f: AffineTransform(shear=f), (2,))

    def __stretch(self, X, Y):
        print("Stretch")
        return self.__transform(X, Y, self.__log2StretchFactors, lambda f: AffineTransform(scale=(2**f, 2**f)), (3, 4))

    def __transform(self, X, Y, factors, transformFromFactor, yIndexOffsets):
        imageToOriginTransform = self.__imageToOriginTransform()
        imageFromOriginTransform = self.__imageFromOriginTransform()

        inverseTransforms = [
            (imageToOriginTransform + (transformFromFactor(factor) + imageFromOriginTransform)).inverse
            for factor in factors
        ]

        XTransformed = [[img_as_ubyte(warp(x, inverseTransform)) for inverseTransform in inverseTransforms] for x in X]

        YTransformed = []
        yLength = self.__baseDatasetLoader.dataPointShape()[1][0]
        for y in Y:
            yTransformed = []
            for factor in factors:
                yCopy = np.array(y)
                for yIndexOffset in yIndexOffsets:
                    yCopy[yIndexOffset + yLength] = factor
                yTransformed.append(yCopy)
            YTransformed.append(yTransformed)

        return np.array(XTransformed), np.array(YTransformed)

    def __imageToOriginTransform(self):
        centre = self.__getCentre()
        minusCentre = tuple(map(lambda x: -x, centre))
        return AffineTransform(translation=minusCentre)

    def __imageFromOriginTransform(self):
        centre = self.__getCentre()
        return AffineTransform(translation=centre)

    def __getCentre(self):
        imageSize = self.dataPointShape()[0][0:2]
        imageCentre = map(lambda x: x / 2 - 0.5, imageSize)
        return tuple(imageCentre)

    def dataPointShape(self):
        oldXShape, oldYShape = self.__baseDatasetLoader.dataPointShape()
        paddingExtra = tuple(map(sum, self.__padding))
        newXShape = (oldXShape[0] + paddingExtra[0], oldXShape[1] + paddingExtra[1]) + oldXShape[2:]
        newYShape = (oldYShape[0] + 5,)
        return newXShape, newYShape
