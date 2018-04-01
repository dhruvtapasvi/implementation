from abc import ABCMeta, abstractclassmethod
import numpy as np


class InterpolateDatasetLoader(metaclass=ABCMeta):
    @abstractclassmethod
    def loadInterpolationData(self) -> (np.ndarray, np.ndarray):
        """
        Should produce a dataset between which you can interpolate
        :return: Two arrays
        First array dimensions: no(examples) * 4 * ...imagedim... . Second index explained as follows:
        0: Left for interpolation
        1: Right for interpolation
        2: Correct for interpolation
        3: Incorrect for interpolation, control result
        Second array dimensions: no(examples) * 4 * ...labeldim... . Second index explained as above
        """
        raise NotImplementedError

    def dataPointShape(self):
        return self.loadInterpolationData()[0][0, 0].shape, self.loadInterpolationData()[1][0, 0].shape
