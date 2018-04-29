from evaluation.metric.Metric import Metric
from evaluation.metric.MetricResult import MetricResult
import numpy as np
from keras.backend import variable, binary_crossentropy, sum, eval, get_session, set_session
import tensorflow as tf


BLOCK_SIZE = 1000


class BinaryCrossEntropy(Metric):
    def compute(self, first: np.ndarray, second: np.ndarray) -> MetricResult:
        oldSession = get_session()

        with tf.Session() as sess:
            set_session(sess)
            firstFlattened = variable(first.reshape((first.shape[0], -1)))
            secondFlattened = variable(second.reshape((second.shape[0], -1)))
            result = eval(sum(binary_crossentropy(firstFlattened, secondFlattened), axis=-1))

        set_session(oldSession)


        # numElements = len(first)
        # result = np.zeros((numElements,), dtype='float64')
        #
        # for i in range(numElements // BLOCK_SIZE):
        #     startIndex = BLOCK_SIZE * i
        #     endIndex = min(BLOCK_SIZE * (i+1), numElements)
        #
        #     firstBlock = first[startIndex:endIndex]
        #     secondBlock = second[startIndex:endIndex]
        #
        #     firstBlockFlattened = variable(firstBlock.reshape((len(firstBlock), -1)))
        #     secondBlockFlattened = variable(secondBlock.reshape((len(secondBlock), -1)))
        #
        #     result[startIndex:endIndex] = eval(sum(binary_crossentropy(firstBlockFlattened, secondBlockFlattened), axis=-1))

        return MetricResult(result)
