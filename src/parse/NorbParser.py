from parse.NorbParsed import UnderlyingType, NorbParsed, translateUnderlyingTypeToNumpyType
import numpy as np


class NorbParser:
    _MIN_NUM_DIMENSIONS = 3

    def _readInts(self, fileHandle, numInts):
        return np.fromfile(fileHandle, dtype=np.uint32, count=numInts)

    def _readUnderlyingType(self, fileHandle):
        return UnderlyingType(self._readInts(fileHandle, 1)[0])

    def _readNumDimensions(self, fileHandle):
        return self._readInts(fileHandle, 1)[0]

    def _readDimensions(self, fileHandle, numDimensions: int):
        return (self._readInts(fileHandle, max(numDimensions, NorbParser._MIN_NUM_DIMENSIONS)))[0:numDimensions]

    def _readData(self, fileHandle, underlyingType: UnderlyingType, dimensions: np.ndarray):
        return np.fromfile(fileHandle, dtype=translateUnderlyingTypeToNumpyType(underlyingType))\
            .reshape(tuple(dimensions))

    def parse(self, fileHandle) -> NorbParsed:
        norbParsed = NorbParsed()
        norbParsed.underlyingType = self._readUnderlyingType(fileHandle)
        norbParsed.numDimensions = self._readNumDimensions(fileHandle)
        norbParsed.dimensions = self._readDimensions(fileHandle, norbParsed.numDimensions)
        norbParsed.data = self._readData(fileHandle, norbParsed.underlyingType, norbParsed.dimensions)
        return norbParsed
