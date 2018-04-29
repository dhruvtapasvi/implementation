from evaluation.metric.Metric import Metric
from evaluation.metric.MetricResult import MetricResult
import numpy as np
from keras.backend import variable, binary_crossentropy, sum, eval


BLOCK_SIZE = 1000


class BinaryCrossEntropy(Metric):
    def compute(self, first: np.ndarray, second: np.ndarray) -> MetricResult:
        numElements = len(first)
        result = np.zeros((numElements,), dtype='float64')

        for i in range(numElements/BLOCK_SIZE):
            startIndex = BLOCK_SIZE * i
            endIndex = min(BLOCK_SIZE * (i+1), numElements)

            firstBlock = first[startIndex:endIndex]
            secondBlock = second[startIndex:endIndex]

            firstBlockFlattened = variable(firstBlock.reshape((len(firstBlock), -1)))
            secondBlockFlattened = variable(secondBlock.reshape((len(secondBlock), -1)))

            result[startIndex:endIndex] = eval(sum(binary_crossentropy(firstBlockFlattened, secondBlockFlattened), axis=-1))

        return MetricResult(result)
