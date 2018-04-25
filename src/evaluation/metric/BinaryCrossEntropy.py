from evaluation.metric.Metric import Metric
from evaluation.metric.MetricResult import MetricResult
import numpy as np
import math
from keras.metrics import binary_crossentropy


class BinaryCrossEntropy(Metric):
    def compute(self, first: np.ndarray, second: np.ndarray) -> MetricResult:
        firstFlattened = first.reshape((first.shape[0], -1))
        secondFlattened = second.reshape((second.shape[0], -1))
        totalNumPixels = len(firstFlattened[0])
        return MetricResult(totalNumPixels * binary_crossentropy(firstFlattened, secondFlattened))
