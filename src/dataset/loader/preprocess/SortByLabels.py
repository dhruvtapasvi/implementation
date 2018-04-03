import numpy as np

from dataset.loader.DatasetLoader import DatasetLoader


class SortByLabels(DatasetLoader):
    def __init__(self, baseDatasetLoader: DatasetLoader):
        self.__baseDatasetLoader = baseDatasetLoader

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        (XTrain, YTrain), (XValidation, YValidation), (XTest, YTest) = self.__baseDatasetLoader.loadData()
        XTrain, YTrain = self.__sort(XTrain, YTrain)
        XValidation, YValidation = self.__sort(XValidation, YValidation)
        XTest, YTest = self.__sort(XTest, YTest)
        return (XTrain, YTrain), (XValidation, YValidation), (XTest, YTest)

    def __sort(self, X: np.ndarray, Y: np.ndarray):
        YLabelFormat = Y if len(Y.shape) == 2 else np.array([Y]).T
        sortPerm = np.lexsort(np.transpose(YLabelFormat), axis=0)
        return X[sortPerm], Y[sortPerm]

    def dataPointShape(self):
        return self.__baseDatasetLoader.dataPointShape()
