from abc import ABCMeta, abstractclassmethod
import numpy as np
from typing import List
from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset


class InterpolateDatasetLoader(metaclass=ABCMeta):
    @abstractclassmethod
    def loadInterpolationData(self) -> List[InterpolateSubdataset]:
        """
        Should produce a dataset between which you can interpolate
        :return: Each interpolate subdataset has a different interpolation factor
        e.g. one interpolate subdataset for angle interpolation and another for size interpolation
        """
        raise NotImplementedError

    def dataPointShape(self):
        return self.loadInterpolationData()[0].xLeft.shape, self.loadInterpolationData()[0].yLeft.shape
