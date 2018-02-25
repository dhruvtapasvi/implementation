import unittest

import numpy as np

from datasets.assemble.NorbAssembler import NorbAssembler


class Test_NorbAssembler(unittest.TestCase):
    def setUp(self):
        self._norbAssember = NorbAssembler()

    def test_stereoPairsAreSeparated_AndCategoriesUpdated(self):
        firstImage = np.arange(0, 6, 1, np.int8).reshape(2, 3)
        secondImage = np.arange(6, 12, 1, np.int8).reshape(2, 3)
        thirdImage = np.arange(12, 18, 1, np.int8).reshape(2, 3)
        fourthImage = np.arange(18, 24, 1, np.int8).reshape(2, 3)
        images = np.array([
            [firstImage, secondImage],
            [thirdImage, fourthImage]
        ])
        firstLabel = np.arange(24, 30, 1, np.int8)
        secondLabel = np.arange(30, 36, 1, np.int8)
        labels = np.array([firstLabel, secondLabel])

        newImages, newLabels = self._norbAssember.assemble(images, labels)

        expectedImages = np.array([firstImage, secondImage, thirdImage, fourthImage])
        expectedLabels = np.array([
            np.concatenate((np.array([0]), firstLabel)),
            np.concatenate((np.array([1]), firstLabel)),
            np.concatenate((np.array([0]), secondLabel)),
            np.concatenate((np.array([1]), secondLabel))
        ])

        np.testing.assert_array_equal(expectedImages, newImages)
        np.testing.assert_array_equal(expectedLabels, newLabels)

if __name__ == "__main__":
    unittest.main()
