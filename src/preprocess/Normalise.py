from preprocess.Preprocessor import Preprocessor
import numpy as np


class Normalise(Preprocessor):
    def preprocess(self, trainData: np.ndarray, testData: np.ndarray) -> (np.ndarray, np.ndarray):
        trainDataStd = np.std(trainData, 0)
        # TODO: What about divide by zero?
        return trainData / trainDataStd, testData / trainDataStd
