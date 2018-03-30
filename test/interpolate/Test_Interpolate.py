import unittest
import numpy as np
from interpolate.Interpolate import Interpolate


class Test_Interpolate(unittest.TestCase):
    def setUp(self):
        self.__interpolate = Interpolate()

    def test_Interpolation(self):
        left = np.array([np.full((4,), i) for i in range(10)])
        right = np.array([np.full((4,), i) for i in range(40, 50)])

        interpolatedResult = self.__interpolate.interpolateAll(left, right, 4).astype(np.int)

        expectedResult = np.array([
            [np.full((4,), i+j) for j in range(0, 50, 10)]
            for i in range(10)
        ])
        np.testing.assert_array_equal(expectedResult, interpolatedResult)
