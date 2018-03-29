from metric.Metric import Metric
import numpy as np
import math


class SquaredError(Metric):
    def __init__(self, variance):
        self.__variance = variance

    def compute(self, first: np.ndarray, second: np.ndarray):
        firstFlattened = first.reshape((first.shape[0], -1))
        secondFlattened = second.reshape((second.shape[0], -1))

        squaredErrorPixelwise = 0.5 * np.square(firstFlattened - secondFlattened) / self.__variance
        varianceErrorPixelwise = 0.5 * math.log(self.__variance)
        constantErrorPixelwise = 0.5 * math.log(2.0 * math.pi)
        totalErrorPixelwise = squaredErrorPixelwise + varianceErrorPixelwise + constantErrorPixelwise

        return np.sum(totalErrorPixelwise, axis=1)
