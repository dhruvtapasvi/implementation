import numpy as np

from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.loader.DatasetLoader import DatasetLoader
from dataset.loader.preprocess.SortByLabels import SortByLabels
from dataset.process.FilterDatasetLabelPredicate import FilterDatasetLabelPredicate
import dataset.info.NorbInfo as norbInfo


class NorbInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, norbLoader: DatasetLoader):
        self.__norbLoader = SortByLabels(norbLoader)

    def loadInterpolationData(self) -> (np.ndarray, np.ndarray):
        _, _, (xTest, yTest) = self.__norbLoader.loadData()
        xSplitByElevation, ySplitByElevation = self.__splitByElevation(xTest, yTest)
        xSplitByAzimuth, ySplitByAzimuth = self.__splitByAzimuth(xTest, yTest)
        return np.concatenate((xSplitByElevation, xSplitByAzimuth)), np.concatenate((ySplitByElevation, ySplitByAzimuth))

    def __splitByElevation(self, X, Y):
        return self.__split(norbInfo.NorbLabelIndex.ELEVATION.value, norbInfo.NORB_ELEVATION_FACTORS, X, Y)

    def __splitByAzimuth(self, X, Y):
        return self.__split(norbInfo.NorbLabelIndex.AZIMUTH.value, norbInfo.NORB_AZIMUTH_FACTORS, X, Y)

    def __split(self, index, factors, X, Y):
        labelFilter = FilterDatasetLabelPredicate()
        perFactorFilter = lambda factor: (labelFilter.filter(X, Y, lambda y: y[index] == factor))
        filteredByFactor = tuple(map(perFactorFilter, factors))
        xByFactor, yByFactor = zip(*filteredByFactor)
        print(np.array(xByFactor).shape)
        print(np.array(yByFactor).shape)
        return np.array(xByFactor).swapaxes(0, 1), np.array(yByFactor).swapaxes(0, 1)

    def dataPointShape(self):
        return self.__norbLoader.dataPointShape()
