import unittest

import numpy as np

from datasets.process.FilterDatasetLabelPredicate import FilterDatasetLabelPredicate


class Test_FilterDatasetLabelPredicate(unittest.TestCase):
    def setUp(self):
        self.__filterDataset = FilterDatasetLabelPredicate()

    def test_filtersScalarConditionCorrectly(self):
        tenToThirtyStep2 = np.arange(10, 30, 2, np.int32)
        zeroToTen = np.arange(0, 10, 1, np.int32)
        isEven = lambda x: x % 2 == 0

        (filteredTenToThirtyStep2, filteredZeroToTen) = self.__filterDataset.filter(tenToThirtyStep2, zeroToTen, isEven)

        tenToThirtyStep4 = np.arange(10, 30, 4, np.int32)
        zeroToTenStep2 = np.arange(0, 10, 2, np.int32)
        np.testing.assert_array_equal(tenToThirtyStep4, filteredTenToThirtyStep2)
        np.testing.assert_array_equal(zeroToTenStep2, filteredZeroToTen)

    def test_filtersVectorConditionCorrectly(self):
        zeroToThirty2D = np.arange(0, 30, 1, np.int32).reshape((10, 3))
        zeroToTen = np.arange(0, 10, 1)
        isEven = lambda x: x % 2 == 0
        secondElementIsEven = lambda row: isEven(row[1])

        (filteredZeroToTen, filteredZeroToThirty2D) =\
            self.__filterDataset.filter(zeroToTen, zeroToThirty2D, secondElementIsEven)

        oneToTenStep2 = np.arange(1, 10, 2)
        expectedFilteredZeroToThirty2D = np.array([[i, i+1, i+2] for i in range(3, 30, 6)])
        np.testing.assert_array_equal(oneToTenStep2, filteredZeroToTen)
        np.testing.assert_array_equal(expectedFilteredZeroToThirty2D, filteredZeroToThirty2D)

    def test_splitsScalarConditionCorrectly(self):
        tenToThirtyStep2 = np.arange(10, 30, 2, np.int32)
        zeroToTen = np.arange(0, 10, 1, np.int32)
        isEven = lambda x: x % 2 == 0

        (splitTrueTenToThirtyStep2, splitTrueZeroToTen), (splitFalseTenToThirtyStep2, splitFalseZeroToTen) =\
            self.__filterDataset.split(tenToThirtyStep2, zeroToTen, isEven)

        tenToThirtyStep4 = np.arange(10, 30, 4, np.int32)
        zeroToTenStep2 = np.arange(0, 10, 2, np.int32)
        np.testing.assert_array_equal(tenToThirtyStep4, splitTrueTenToThirtyStep2)
        np.testing.assert_array_equal(zeroToTenStep2, splitTrueZeroToTen)

        twelveToThirtyStep4 = np.arange(12, 30, 4, np.int32)
        oneToTenStep2 = np.arange(1, 10, 2, np.int32)
        np.testing.assert_array_equal(twelveToThirtyStep4, splitFalseTenToThirtyStep2)
        np.testing.assert_array_equal(oneToTenStep2, splitFalseZeroToTen)

    def test_splitsVectorConditionCorrectly(self):
        zeroToThirty2D = np.arange(0, 30, 1, np.int32).reshape((10, 3))
        zeroToTen = np.arange(0, 10, 1)
        isEven = lambda x: x % 2 == 0
        secondElementIsEven = lambda row: isEven(row[1])

        (splitTrueZeroToTen, splitTrueZeroToThirty2D), (splitFalseZeroToTen, splitFalseZeroToThirty2D) =\
            self.__filterDataset.split(zeroToTen, zeroToThirty2D, secondElementIsEven)

        oneToTenStep2 = np.arange(1, 10, 2)
        expectedSplitTrueZeroToThirty2D = np.array([[i, i+1, i+2] for i in range(3, 30, 6)])
        np.testing.assert_array_equal(oneToTenStep2, splitTrueZeroToTen)
        np.testing.assert_array_equal(expectedSplitTrueZeroToThirty2D, splitTrueZeroToThirty2D)

        zeroToTenStep2 = np.arange(0, 10, 2)
        expectedSplitFalseZeroToThirty2D = np.array([[i, i+1, i+2] for i in range(0, 30, 6)])
        np.testing.assert_array_equal(zeroToTenStep2, splitFalseZeroToTen)
        np.testing.assert_array_equal(expectedSplitFalseZeroToThirty2D, splitFalseZeroToThirty2D)

if __name__ == "__main__":
    unittest.main()
