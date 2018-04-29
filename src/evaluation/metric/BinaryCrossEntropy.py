from evaluation.metric.Metric import Metric
from evaluation.metric.MetricResult import MetricResult
import numpy as np


EPSILON = 1e-7 # Use same epsilon as Keras backend default


class BinaryCrossEntropy(Metric):
    def compute(self, first: np.ndarray, second: np.ndarray) -> MetricResult:
        firstFlattened = first.reshape((len(first), -1))
        secondFlattened = second.reshape((len(second), -1))
        secondClipped = np.clip(secondFlattened, EPSILON, 1 - EPSILON)
        binaryCrossentropy = -firstFlattened * np.log(secondClipped) - (1 - firstFlattened) * np.log(1 - secondClipped)
        result = np.sum(binaryCrossentropy, axis=1)
        return MetricResult(result)
