from abc import ABCMeta, abstractclassmethod
import numpy as np
from evaluation.metric.MetricResult import MetricResult


class Metric(metaclass=ABCMeta):
    @abstractclassmethod
    def compute(self, first: np.ndarray, second: np.ndarray) -> MetricResult:
        raise NotImplementedError
