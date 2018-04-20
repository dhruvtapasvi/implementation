from abc import ABCMeta, abstractclassmethod
import numpy as np


class Metric(metaclass=ABCMeta):
    @abstractclassmethod
    def compute(self, first: np.ndarray, second: np.ndarray):
        raise NotImplementedError
