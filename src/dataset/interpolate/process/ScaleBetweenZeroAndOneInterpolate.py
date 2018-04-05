import numpy as np
from typing import List

from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset


class ScaleBetweenZeroAndOneInterpolate(InterpolateDatasetLoader):
    def __init__(self, baseInterpolateLoader: InterpolateDatasetLoader, minimum, maximum):
        self.__baseInterpolateLoader = baseInterpolateLoader
        self.__minimum = float(minimum)
        self.__range = float(maximum - minimum)

    def loadInterpolationData(self) -> List[InterpolateSubdataset]:
        return [
            InterpolateSubdataset(
                interpolateSubdataset.interpolatedFactorName,
                (interpolateSubdataset.xLeft.astype("float64") / self.__range + self.__minimum, interpolateSubdataset.yLeft),
                (interpolateSubdataset.xRight.astype("float64") / self.__range + self.__minimum, interpolateSubdataset.yRight),
                (interpolateSubdataset.xCentre.astype("float64") / self.__range + self.__minimum, interpolateSubdataset.yCentre)
                    if interpolateSubdataset.centreIsSpecified() else None,
                (interpolateSubdataset.xOutside.astype("float64") / self.__range + self.__minimum, interpolateSubdataset.yOutside)
                    if interpolateSubdataset.outsideIsSpecified() else None,
            ) for interpolateSubdataset in self.__baseInterpolateLoader.loadInterpolationData()
        ]

    def dataPointShape(self):
        return self.__baseInterpolateLoader.dataPointShape()
