from abc import ABCMeta, abstractclassmethod
import numpy as np


class DatasetLoader(metaclass=ABCMeta):
    @abstractclassmethod
    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        raise NotImplementedError

    def dataPointShape(self):
        return self.loadData()[0][0].shape, self.loadData()[0][1].shape
