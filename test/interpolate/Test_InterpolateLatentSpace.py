import unittest
import numpy as np
from model.Autoencoder import Autoencoder
from interpolate.InterpolateLatentSpace import InterpolateLatentSpace


class MockEncoder():
    def predict_on_batch(self, x):
        return np.sqrt(x[:, :2]).astype(np.int)


class MockDecoder():
    def predict_on_batch(self, y):
        return (np.square(np.concatenate((y, y), axis=1)) + 0.5).astype(np.int)


class MockAutoencoder(Autoencoder):
    def autoencoder(self):
        pass

    def encoder(self):
        return MockEncoder()

    def decoder(self):
        return MockDecoder()


class Test_InterpolateLatentSpace(unittest.TestCase):
    def setUp(self):
        self.__autoencoder = MockAutoencoder()
        self.__interpolate = InterpolateLatentSpace(self.__autoencoder)

    def test_InterpolationInLatentSpace(self):
        left = np.square([np.full((4,), i) for i in range(10)])
        right = np.square([np.full((4,), i) for i in range(10, 20)])

        interpolatedResult = self.__interpolate.interpolateAll(left, right, 5)

        expectedResult = np.square([
            [np.full((4,), i + j) for j in range(0, 10 + 2, 2)]
            for i in range(10)
        ])
        np.testing.assert_array_equal(expectedResult, interpolatedResult)
