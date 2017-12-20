from datasets.DatasetLoader import DatasetLoader
import numpy as np
from keras.datasets import mnist

class MnistLoader(DatasetLoader):
    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        return mnist.load_data()
