import unittest

import numpy as np

from dataset.loader.DatasetLoader import DatasetLoader
from dataset.loader.preprocess.SortByLabels import SortByLabels


class MockDatasetLoader_SingleLabelPerExample(DatasetLoader):
    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        XTrain = np.array([np.full((4,), i) for i in range(6)])
        YTrain = np.arange(6, 0, -1)
        XVal = np.array([np.full((4,), i) for i in range(6, 8)])
        YVal = np.arange(8, 6, -1)
        XTest = np.array([np.full((4,), i) for i in range(8, 10)])
        YTest = np.arange(10, 8, -1)
        return (XTrain, YTrain), (XVal, YVal), (XTest, YTest)


class MockDatasetLoader_MultipleLabelsPerExample(DatasetLoader):
    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        XTrain = np.array([np.full((4,), i) for i in range(6)])
        YTrain = np.array([[1, i] for i in range(6, 0, -1)])
        XVal = np.array([np.full((4,), i) for i in range(6, 8)])
        YVal = np.array([[1, i] for i in range(8, 6, -1)])
        XTest = np.array([np.full((4,), i) for i in range(8, 10)])
        YTest = np.array([[1, i] for i in range(10, 8, -1)])
        return (XTrain, YTrain), (XVal, YVal), (XTest, YTest)


class MockDatasetLoader_SingleLabelPerExample_EncapsulatedAsMultiple(DatasetLoader):
    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        XTrain = np.array([np.full((4,), i) for i in range(6)])
        YTrain = np.array([[i] for i in range(6, 0, -1)])
        XVal = np.array([np.full((4,), i) for i in range(6, 8)])
        YVal = np.array([[i] for i in range(8, 6, -1)])
        XTest = np.array([np.full((4,), i) for i in range(8, 10)])
        YTest = np.array([[i] for i in range(10, 8, -1)])
        return (XTrain, YTrain), (XVal, YVal), (XTest, YTest)


class Test_SortByLabels(unittest.TestCase):
    def test_SingleLabelPerExample(self):
        self.sortsByLabelsTest(MockDatasetLoader_SingleLabelPerExample())

    def test_MultipleLabelsPerExample(self):
        self.sortsByLabelsTest(MockDatasetLoader_MultipleLabelsPerExample())

    def test_SingleLabelPerExample_EncapsulatedAsMultiple(self):
        self.sortsByLabelsTest(MockDatasetLoader_SingleLabelPerExample_EncapsulatedAsMultiple())

    def sortsByLabelsTest(self, mockDatasetLoader):
        mockData = mockDatasetLoader.loadData()
        sortByLabels = SortByLabels(mockDatasetLoader)

        sortedMockData = sortByLabels.loadData()

        for i in range(3):
            for j in range(2):
                expectedSortedData = np.flip(mockData[i][j], 0)
                np.testing.assert_array_equal(expectedSortedData, sortedMockData[i][j])
