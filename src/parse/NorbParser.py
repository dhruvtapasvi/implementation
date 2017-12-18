from parse.NorbParsed import UnderlyingType, NorbParsed, translateUnderlyingTypeToNumpyType
import numpy as np


class NorbParser:
    _MIN_NUM_DIMENSIONS = 3

    def __init__(self, fileHandle):
        self._fileHandle = fileHandle
        self._norbParsed = NorbParsed()

    def _readInts(self, numInts):
        return np.fromfile(self._fileHandle, dtype=np.uint32, count=numInts)

    def _readType(self):
        self._norbParsed.type = UnderlyingType(self._readInts(1)[0])

    def _readNumDimensions(self):
        self._norbParsed.numDimensions = self._readInts(1)[0]

    def _readDimensions(self):
        self._norbParsed.dimensions = \
            (self._readInts(max(self._norbParsed.numDimensions, NorbParser._MIN_NUM_DIMENSIONS)))[0:self._norbParsed.numDimensions]

    def _readData(self):
        self._norbParsed.data = np.fromfile(
            self._fileHandle,
            dtype=translateUnderlyingTypeToNumpyType(self._norbParsed.type)
        ).reshape(tuple(self._norbParsed.dimensions))

    def parse(self) -> NorbParsed:
        self._readType()
        self._readNumDimensions()
        self._readDimensions()
        self._readData()
        return self._norbParsed
