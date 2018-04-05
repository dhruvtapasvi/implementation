from typing import List

from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset
from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.interpolate.process.InstancesInterpolateLoader import InstancesInterpolateLoader
import dataset.info.MnistInfo as mnistInfo
from dataset.loader.DatasetLoader import DatasetLoader


class MnistInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, mnistLoader: DatasetLoader):
        self.__mnistLoader = InstancesInterpolateLoader(mnistLoader, mnistInfo.MNIST_INTERPOLATION_INSTANCES)

    def loadInterpolationData(self) -> List[InterpolateSubdataset]:
        return self.__mnistLoader.loadInterpolationData()

    def dataPointShape(self):
        return mnistInfo.MNIST_IMAGE_DIMENSIONS, mnistInfo.MNIST_LABEL_DIMENSIONS
