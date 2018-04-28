from evaluation.metric.Metric import Metric
from evaluation.metric.MetricResult import MetricResult
import numpy as np
from keras.backend import variable, binary_crossentropy, sum, eval


class BinaryCrossEntropy(Metric):
    def compute(self, first: np.ndarray, second: np.ndarray) -> MetricResult:
        firstFlattened = variable(first.reshape((first.shape[0], -1)))
        secondFlattened = variable(second.reshape((second.shape[0], -1)))
        result = eval(sum(binary_crossentropy(firstFlattened, secondFlattened), axis=-1))
        return MetricResult(result)
