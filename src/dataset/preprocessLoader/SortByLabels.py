from dataset.DatasetLoader import DatasetLoader
import numpy as np


class SortByLabels(DatasetLoader):
    def __init__(self, baseDatasetLoader: DatasetLoader):
        self.__baseDatasetLoader = baseDatasetLoader

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        print("Sorting...")
        (XTrain, YTrain), (XValidation, YValidation), (XTest, YTest) = self.__baseDatasetLoader.loadData()
        XTrain, YTrain = self.__sort(XTrain, YTrain)
        XValidation, YValidation = self.__sort(XValidation, YValidation)
        XTest, YTest = self.__sort(XTest, YTest)
        print("Sorted!")
        return (XTrain, YTrain), (XValidation, YValidation), (XTest, YTest)

    def __sort(self, X: np.ndarray, Y: np.ndarray):
        sortPerm = np.lexsort(np.transpose(Y), axis=0)
        return X[sortPerm], Y[sortPerm]
