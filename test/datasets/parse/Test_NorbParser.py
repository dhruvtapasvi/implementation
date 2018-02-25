import os
import unittest

import numpy as np
from parse.NorbParser import NorbParser

from datasets.parse.NorbParsed import NorbParsed, UnderlyingType

mockFilePath = "./Test_NorbParser_Scratch"


def writeFileContentsToMockFile(fileContents):
    with open(mockFilePath, 'w+') as mockFile:
        fileContents.tofile(mockFile)


def parseMockFile() -> NorbParsed:
    norbParser = NorbParser()
    with open(mockFilePath, 'r') as mockFile:
        return norbParser.parse(mockFile)


def deleteTestFile():
    os.remove(mockFilePath)


def assertNorbParsedEquality(
        testCase: unittest.TestCase,
        underlyingType: UnderlyingType,
        numDimensions: int,
        dimensions: np.ndarray,
        data: np.ndarray,
        norbParsed: NorbParsed):
    testCase.assertEqual(underlyingType, norbParsed.underlyingType)
    testCase.assertEqual(numDimensions, norbParsed.numDimensions)
    np.testing.assert_array_equal(dimensions, norbParsed.dimensions)
    np.testing.assert_array_equal(data, norbParsed.data)


class Test_NorbParser_OneDimensionFile(unittest.TestCase):
    def setUp(self):
        fileContents = np.array([UnderlyingType.INTEGER.value, 1, 1, 1, 1, 1], dtype=np.uint32)
        writeFileContentsToMockFile(fileContents)
        self._norbParsed = parseMockFile()

    def test_FileIsCorrectlyParsed(self):
        assertNorbParsedEquality(
            self,
            UnderlyingType.INTEGER,
            1,
            np.array([1], dtype=np.uint32),
            np.array([1], dtype=np.uint32),
            self._norbParsed
        )

    def tearDown(self):
        deleteTestFile()


class Test_NorbParser_ThreeDimensionFile(unittest.TestCase):
    def setUp(self):
        fileContents = np.array([UnderlyingType.BYTE.value, 3, 2, 2, 2, 1, 1], dtype=np.uint32)
        writeFileContentsToMockFile(fileContents)
        self._norbParsed = parseMockFile()

    def test_FileIsCorrectlyParsed(self):
        assertNorbParsedEquality(
            self,
            UnderlyingType.BYTE,
            3,
            np.array([2, 2, 2], dtype=np.uint32),
            np.array([
                [[1, 0], [0, 0]],  # 1 at the beginning since little endian!
                [[1, 0], [0, 0]]
            ], dtype=np.uint8),
            self._norbParsed
        )

    def tearDown(self):
        deleteTestFile()


class Test_NorbParser_FourDimensionFile(unittest.TestCase):
    def setUp(self):
        fileContents = np.array(
            [UnderlyingType.INTEGER.value, 5, 2, 1, 1, 2, 2, 1, 2, 3, 4, 5, 6, 7, 8],
            dtype=np.uint32
        )
        writeFileContentsToMockFile(fileContents)
        self._norbParsed = parseMockFile()

    def test_FileIsCorrectlyParsed(self):
        assertNorbParsedEquality(
            self,
            UnderlyingType.INTEGER,
            5,
            np.array([2, 1, 1, 2, 2], dtype=np.uint32),
            np.array([
                [[[[1, 2], [3, 4]]]],
                [[[[5, 6], [7, 8]]]],
            ], dtype=np.uint32),
            self._norbParsed
        )

    def tearDown(self):
        deleteTestFile()


if __name__ == "__main__":
    unittest.main()
