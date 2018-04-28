import numpy as np
from skimage.transform import AffineTransform, warp
from skimage.util import pad, img_as_ubyte
from typing import List

from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.loader.DatasetLoader import DatasetLoader
from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset


class CreateTransformedInterpolateData(InterpolateDatasetLoader):
    def __init__(
            self,
            baseDatasetLoader: DatasetLoader,
            padding,
            defaultRotationFactor,
            defaultShearFactor,
            defaultLog2StretchFactor,
            rotationFactors,
            shearFactors,
            log2StretchFactors):
        #  Factors have (left, right, centre [optional], outside [optional]) layout
        self.__baseDatasetLoader = baseDatasetLoader
        self.__padding = padding
        self.__rotationFactors = rotationFactors
        self.__defaultRotationFactor = defaultRotationFactor
        self.__shearFactors = shearFactors
        self.__defaultShearFactor = defaultShearFactor
        self.__log2StretchFactors = log2StretchFactors
        self.__defaultLog2StretchFactor = defaultLog2StretchFactor

    def loadInterpolationData(self) -> List[InterpolateSubdataset]:
        _, _, (xTest, yTest) = self.__baseDatasetLoader.loadData()
        xTestPadded = self.__pad(xTest)
        yTestLabelled = self.__label(yTest)
        return [f(xTestPadded, yTestLabelled) for f in [self.__rotate, self.__shear, self.__stretch]]

    def __pad(self, X: np.ndarray):
        XPadded = [pad(x, self.__padding, 'constant') for x in X]
        return np.array(XPadded)

    def __label(self, Y: np.ndarray):
        YLabelled = [
            (index,) + tuple(y) + (self.__defaultRotationFactor, self.__defaultShearFactor, self.__defaultLog2StretchFactor, self.__defaultLog2StretchFactor)
            for index, y in enumerate(Y)
        ]
        return np.array(YLabelled)

    def __rotate(self, X, Y) -> InterpolateSubdataset:
        return self.__transform(X, Y, "ROTATION", self.__rotationFactors, lambda f: AffineTransform(rotation=f, shear=self.__defaultShearFactor, scale=(2**self.__defaultLog2StretchFactor, 2**self.__defaultLog2StretchFactor)), (1,))

    def __shear(self, X, Y) -> InterpolateSubdataset:
        return self.__transform(X, Y, "SHEAR", self.__shearFactors, lambda f: AffineTransform(shear=f, rotation=self.__defaultRotationFactor, scale=(2**self.__defaultLog2StretchFactor, 2**self.__defaultLog2StretchFactor)), (2,))

    def __stretch(self, X, Y) -> InterpolateSubdataset:
        return self.__transform(X, Y, "STRETCH", self.__log2StretchFactors, lambda f: AffineTransform(scale=(2**f, 2**f), rotation=self.__defaultRotationFactor, shear=self.__defaultShearFactor), (3, 4))

    def __transform(self, X, Y, interpolationFactorName, factors, transformFromFactor, yIndexOffsets):
        imageToOriginTransform = self.__imageToOriginTransform()
        imageFromOriginTransform = self.__imageFromOriginTransform()
        yLength = self.__baseDatasetLoader.dataPointShape()[1][0]

        def performTransform(factor):
            inverseTransform = (imageToOriginTransform + (transformFromFactor(factor) + imageFromOriginTransform)).inverse
            XTransformed = np.array([img_as_ubyte(warp(x, inverseTransform)) for x in X])
            YTransformed = np.array(Y)
            for y in YTransformed:
                for yIndexOffset in yIndexOffsets:
                    y[yLength + yIndexOffset] = factor
            return XTransformed, YTransformed

        transformed = map(performTransform, factors)
        return InterpolateSubdataset(interpolationFactorName, *transformed)

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
