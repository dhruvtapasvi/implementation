import unittest
import numpy as np

from dataset.DatasetLoader import DatasetLoader
from dataset.preprocessLoader.SortByLabels import SortByLabels


class MockDatasetLoader(DatasetLoader):
    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        XTrain = np.array([np.full((4,), i) for i in range(6)])
        YTrain = np.array([[1, i] for i in range(6, 0, -1)])
        XVal = np.array([np.full((4,), i) for i in range(6, 8)])
        YVal = np.array([[1, i] for i in range(8, 6, -1)])
        XTest = np.array([np.full((4,), i) for i in range(8, 10)])
        YTest = np.array([[1, i] for i in range(10, 8, -1)])
        return (XTrain, YTrain), (XVal, YVal), (XTest, YTest)


class Test_SortByLabels(unittest.TestCase):
    def setUp(self):
        self.__mockDatasetLoader = MockDatasetLoader()
        (self.__mockXTrain, self.__mockYTrain), (self.__mockXVal, self.__mockYVal), (self.__mockXTest, self.__mockYTest) = self.__mockDatasetLoader.loadData()
        self.__sortByLabels = SortByLabels(self.__mockDatasetLoader)

    def test_SortsByLabels(self):
        (XTrainSorted, YTrainSorted), (XValSorted, YValSorted), (XTestSorted, YTestSorted) = self.__sortByLabels.loadData()

        expectedXTrainSorted = np.flip(self.__mockXTrain, 0)
        expectedYTrainSorted = np.flip(self.__mockYTrain, 0)
        expectedXValSorted = np.flip(self.__mockXVal, 0)
        expectedYValSorted = np.flip(self.__mockYVal, 0)
        expectedXTestSorted = np.flip(self.__mockXTest, 0)
        expectedYTestSorted = np.flip(self.__mockYTest, 0)

        np.testing.assert_array_equal(expectedXTrainSorted, XTrainSorted)
        np.testing.assert_array_equal(expectedYTrainSorted, YTrainSorted)
        np.testing.assert_array_equal(expectedXValSorted, XValSorted)
        np.testing.assert_array_equal(expectedYValSorted, YValSorted)
        np.testing.assert_array_equal(expectedXTestSorted, XTestSorted)
        np.testing.assert_array_equal(expectedYTestSorted, YTestSorted)
