import numpy as np


class MetricResult:
    def __init__(self, allValues: np.ndarray):
        self.__allValues = allValues
        self.__mean = np.mean(allValues)
        self.__standardDeviation = np.std(allValues, ddof=1)

    @property
    def allValues(self) -> np.ndarray:
        return self.__allValues

    @property
    def mean(self) -> float:
        return self.__mean

    @property
    def standardDeviation(self) -> float:
        return self.__standardDeviation
