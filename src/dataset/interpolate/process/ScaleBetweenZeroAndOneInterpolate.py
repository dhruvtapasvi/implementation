import numpy as np
from typing import List

from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset


class ScaleBewteenZeroAndOneInterpolate(InterpolateDatasetLoader):
    def __init__(self, baseInterpolateLoader: InterpolateDatasetLoader, maximum, minimum=0):
        self.__baseInterpolateLoader = baseInterpolateLoader
        self.__minimum = float(minimum)
        self.__range = float(maximum - minimum)

    def loadInterpolationData(self):
        xInterpolate, yInterpolate = self.__baseInterpolateLoader.loadInterpolationData()
        xInterpolate = xInterpolate.astype("float64") / self.__range + self.__minimum
        return xInterpolate, yInterpolate

    def dataPointShape(self):
        return self.__baseInterpolateLoader.dataPointShape()