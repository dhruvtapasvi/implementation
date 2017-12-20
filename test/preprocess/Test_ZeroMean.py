import unittest
from preprocess.ZeroMean import ZeroMean
import numpy as np


class Test_ZeroMean(unittest.TestCase):
    def setUp(self):
        self._zeroMean = ZeroMean()

    def test_ZerosArrayIsUnchanged(self):
        trainData = np.zeros((100, 10))
        testData = np.zeros((5, 10))

        processedTrainData, processedTestData = self._zeroMean.preprocess(trainData, testData)

        np.testing.assert_array_equal(trainData, processedTrainData)
        np.testing.assert_array_equal(testData, processedTestData)

    def test_meanComputedOnlyOverTestData(self):
        trainData = np.zeros((100, 10))
        testData = np.arange(0, 50).reshape(5, 10)

        processedTrainData, processedTestData = self._zeroMean.preprocess(trainData, testData)

        np.testing.assert_array_equal(testData, processedTestData)

    def test_meanIsCorrectlySubtracted(self):
        trainData = np.array([
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]
        ])
        testData = np.array([
            [4, 4, 4, 4]
        ])

        processedTrainData, processedTestData = self._zeroMean.preprocess(trainData, testData)

        np.testing.assert_array_equal(np.array([
            [-1, -1, -1, -1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ]), processedTrainData)
        np.testing.assert_array_equal(np.array([[2, 2, 2, 2]]), processedTestData)

if __name__ == "__main__":
    unittest.main()
