import numpy as np

from datasets.preprocess import Preprocessor


class ZeroMean(Preprocessor):
    def preprocess(self, trainData: np.ndarray, testData: np.ndarray) -> (np.ndarray, np.ndarray):
        trainDataMean = np.mean(trainData, 0)
        return trainData - trainDataMean, testData - trainDataMean
