from abc import ABCMeta, abstractclassmethod
import numpy as np


class Preprocessor(metaclass=ABCMeta):
    @abstractclassmethod
    def preprocess(self, trainData: np.ndarray, testData: np.ndarray) -> (np.ndarray, np.ndarray):
        raise NotImplementedError
