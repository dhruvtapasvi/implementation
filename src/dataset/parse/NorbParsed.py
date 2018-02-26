from enum import Enum
import numpy as np


class UnderlyingType(Enum):
    SINGLE = 0x1E3D4C51
    PACKED = 0x1E3D4C52
    DOUBLE = 0x1E3D4C53
    INTEGER = 0x1E3D4C54
    BYTE = 0x1E3D4C55
    SHORT = 0x1E3D4C56


class UnsupportedUnderlyingType(Exception):
    pass


def translateUnderlyingTypeToNumpyType(underlyingType: UnderlyingType) -> np.dtype:
    if underlyingType == UnderlyingType.SINGLE:
        return np.float32
    elif underlyingType == UnderlyingType.DOUBLE:
        return np.float64
    elif underlyingType == UnderlyingType.INTEGER:
        return np.uint32
    elif underlyingType == UnderlyingType.BYTE:
        return np.uint8
    elif underlyingType == UnderlyingType.SHORT:
        return np.uint16
    else:
        # No support for packed
        raise UnsupportedUnderlyingType


class NorbParsed:
    def __init__(self):
        self._underlyingType = None
        self._numDimensions = None
        self._dimensions = None
        self._data = None

    @property
    def underlyingType(self) -> UnderlyingType:
        return self._underlyingType

    @underlyingType.setter
    def underlyingType(self, value: UnderlyingType):
        self._underlyingType = value

    @underlyingType.deleter
    def underlyingType(self):
        del self._underlyingType

    @property
    def numDimensions(self) -> int:
        return self._numDimensions

    @numDimensions.setter
    def numDimensions(self, value: int):
        self._numDimensions = value

    @numDimensions.deleter
    def numDimensions(self):
        del self._numDimensions

    @property
    def dimensions(self) -> np.ndarray:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value: np.ndarray):
        self._dimensions = value

    @dimensions.deleter
    def dimensions(self):
        del self._dimensions

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value

    @data.deleter
    def data(self):
        del self._data
