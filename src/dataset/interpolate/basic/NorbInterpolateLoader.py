import numpy as np

from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.loader.DatasetLoader import DatasetLoader
from dataset.loader.preprocess.SortByLabels import SortByLabels
from dataset.process.FilterDatasetLabelPredicate import FilterDatasetLabelPredicate
import dataset.info.NorbInfo as norbInfo
from typing import List
from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset


class NorbInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, norbLoader: DatasetLoader):
        self.__norbLoader = SortByLabels(norbLoader)

    def loadInterpolationData(self) -> List[InterpolateSubdataset]:
        _, _, (xTest, yTest) = self.__norbLoader.loadData()
        return [f(xTest, yTest) for f in [self.__splitByElevation, self.__splitByAzimuth]]

    def __splitByElevation(self, X, Y) -> InterpolateSubdataset:
        return self.__split(norbInfo.NORB_ELEVATION_NAME, norbInfo.NorbLabelIndex.ELEVATION.value, norbInfo.NORB_ELEVATION_FACTORS, X, Y)

    def __splitByAzimuth(self, X, Y) -> InterpolateSubdataset:
        return self.__split(norbInfo.NORB_AZIMUTH_NAME, norbInfo.NorbLabelIndex.AZIMUTH.value, norbInfo.NORB_AZIMUTH_FACTORS, X, Y)

    def __split(self, interpolationFactorName, index, factors, X, Y) -> InterpolateSubdataset:
        labelFilter = FilterDatasetLabelPredicate()
        filteredByFactor = map(lambda factor: labelFilter.filter(X, Y, lambda y: y[index] == factor), factors)
        interpolateSubdataset = InterpolateSubdataset(interpolationFactorName, *filteredByFactor)
        return interpolateSubdataset

    def dataPointShape(self):
        return self.__norbLoader.dataPointShape()
