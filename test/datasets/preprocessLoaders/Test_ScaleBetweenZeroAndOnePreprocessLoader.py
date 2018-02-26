import unittest
import numpy as np
from datasets.DatasetLoader import DatasetLoader
from datasets.preprocessLoaders.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne


class MockDatasetLoader_NonnegativeData(DatasetLoader):
    def __init__(self, imageDimensions, trainSize, validationSize, testSize):
        self.__imageDimensions = imageDimensions
        self.__trainSize = trainSize
        self.__validationSize = validationSize
        self.__testSize = testSize

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        flattenedImageDimension = np.prod(self.__imageDimensions)

        XTrain = np.array([np.arange(flattenedImageDimension).reshape(self.__imageDimensions) for _ in range(self.__trainSize)])
        YTrain = np.arange(self.__trainSize)
        train = XTrain, YTrain

        XValidation = np.array([np.zeros(self.__imageDimensions) for _ in range(self.__validationSize)])
        YValidation = np.arange(self.__validationSize)
        validation = XValidation, YValidation

        XTest = np.array([np.full(self.__imageDimensions, flattenedImageDimension - 1) for _ in range(self.__testSize)])
        YTest = np.arange(self.__testSize)
        test = XTest, YTest


        return train, validation, test

    def __createXandY(self, size):
        flattenedImageDimension = np.prod(self.__imageDimensions)
        X = np.array([np.arange(flattenedImageDimension).reshape(self.__imageDimensions) for _ in range(size)])
        Y = np.arange(size)
        return X, Y


class Test_ScaleBetweenZeroAndOnePreprocessLoader(unittest.TestCase):
    def setUp(self):
        self.__imageDimensions = (3, 3)
        self.__trainSize = 10
        self.__validationSize = 5
        self.__testSize = 2

    def test_ScalingBetweenZeroAndOneForNonnegativeData(self):
        scaler = ScaleBetweenZeroAndOne(
            MockDatasetLoader_NonnegativeData(self.__imageDimensions, self.__trainSize, self.__validationSize, self.__testSize),
            0,
            np.prod(self.__imageDimensions) - 1)
        (XTrain, YTrain), (XValidation, YValidation), (XTest, YTest) = scaler.loadData()

        liesInRangeZeroToOne = lambda x: 0. <= x <= 1.
        XTrainLiesInRange = np.all([liesInRangeZeroToOne(example) for example in XTrain.flatten()])
        XValidationLiesInRange = np.all([liesInRangeZeroToOne(example) for example in XValidation.flatten()])
        XTestLiesInRange = np.all([liesInRangeZeroToOne(example) for example in XTest.flatten()])

        self.assertTrue(XTrainLiesInRange)
        self.assertTrue(XValidationLiesInRange)
        self.assertTrue(XTestLiesInRange)

        flattenedImageDimension = np.prod(self.__imageDimensions)
        expectedTrainImage = np.arange(flattenedImageDimension).reshape(self.__imageDimensions) / (flattenedImageDimension - 1)
        expectedXTrain = np.array([expectedTrainImage for _ in range(self.__trainSize)])

        expectedValidationImage = np.full(self.__imageDimensions, 0.)
        expectedXValidation = np.array([expectedValidationImage for _ in range(self.__validationSize)])

        expectedTestImage = np.full(self.__imageDimensions, 1.)
        expectedXTest = np.array([expectedTestImage for _ in range(self.__testSize)])

        np.testing.assert_almost_equal(expectedXTrain, XTrain)
        np.testing.assert_almost_equal(expectedXValidation, XValidation)
        np.testing.assert_almost_equal(expectedXTest, XTest)
