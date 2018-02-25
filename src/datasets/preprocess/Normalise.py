import numpy as np

from datasets.preprocess import Preprocessor


class Normalise(Preprocessor):
    def preprocess(self, trainData: np.ndarray, testData: np.ndarray) -> (np.ndarray, np.ndarray):
        trainDataStd = np.std(trainData, 0)
        # TODO: What about divide by zero?
        # TODO: Write test
        return trainData / trainDataStd, testData / trainDataStd
