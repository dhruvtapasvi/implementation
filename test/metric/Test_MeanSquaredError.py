import math
import unittest

import numpy as np

from evaluation.metric.SquaredError import SquaredError


class Test_MeanSquaredError(unittest.TestCase):
    def setUp(self):
        self.__defaultVariance = 1.0
        self.__squaredError = SquaredError(self.__defaultVariance)

    def test_EmptyArraysHaveZeroError(self):
        numImages = 10
        first = np.array([[] for _ in range(numImages)])
        second = np.array([[] for _ in range(numImages)])

        totalSquaredErrorByImage = self.__squaredError.compute(first, second).allValues

        expectedTotalSquaredErrorByImage = np.zeros(numImages)
        np.testing.assert_array_almost_equal(expectedTotalSquaredErrorByImage, totalSquaredErrorByImage)

    def test_UnitVarianceIdenticalImages_HaveOnlyPiError(self):
        imageSize = (10, 2)
        numImages = 10
        firstImages = np.array([np.full(imageSize, i) for i in range(numImages)])
        secondImages = np.array([np.full(imageSize, i) for i in range(numImages)])

        totalSquaredErrorByImage = self.__squaredError.compute(firstImages, secondImages).allValues

        expectedTotalSquaredErrorByImage = np.full((numImages,), 0.5 * np.prod(imageSize) * math.log(2 * math.pi))
        np.testing.assert_array_almost_equal(expectedTotalSquaredErrorByImage, totalSquaredErrorByImage)

    def test_UnitVarianceDifferentImages(self):
        imageSize = (10, 2)
        numImages = 10
        firstImages = np.array([np.full(imageSize, i) for i in range(numImages)])
        secondImages = np.array([np.full(imageSize, i+1) for i in range(numImages)])

        totalSquaredErrorByImage = self.__squaredError.compute(firstImages, secondImages).allValues

        expectedTotalSquaredErrorByImage = np.full((numImages,), 0.5 * np.prod(imageSize) * (math.log(2 * math.pi) + 1))
        np.testing.assert_array_almost_equal(expectedTotalSquaredErrorByImage, totalSquaredErrorByImage)

    def test_NonunitVarianceSameImages(self):
        imageSize = (10, 2)
        numImages = 10
        variance = 0.3
        squaredError = SquaredError(variance)
        firstImages = np.array([np.full(imageSize, i) for i in range(numImages)])
        secondImages = np.array([np.full(imageSize, i) for i in range(numImages)])

        totalSquaredErrorByImage = squaredError.compute(firstImages, secondImages).allValues

        expectedTotalSquaredErrorByImage = np.full((numImages,), 0.5 * np.prod(imageSize) * (math.log(2 * math.pi) + math.log(variance)))
        np.testing.assert_array_almost_equal(expectedTotalSquaredErrorByImage, totalSquaredErrorByImage)

    def test_NonunitVarianceDifferentImages(self):
        imageSize = (10, 2)
        numImages = 10
        variance = 0.3
        squaredError = SquaredError(variance)
        firstImages = np.array([np.full(imageSize, i) for i in range(numImages)])
        secondImages = np.array([np.full(imageSize, i+1) for i in range(numImages)])

        totalSquaredErrorByImage = squaredError.compute(firstImages, secondImages).allValues

        expectedTotalSquaredErrorByImage = np.full((numImages,), 0.5 * np.prod(imageSize) * (math.log(2 * math.pi) + math.log(variance) + 1 / variance))
        np.testing.assert_array_almost_equal(expectedTotalSquaredErrorByImage, totalSquaredErrorByImage)
