from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.loader.DatasetLoader import DatasetLoader
from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset
from typing import List


class InstancesInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, baseDatasetLoader: DatasetLoader, instancesToLoad: List[tuple]):
        self.__baseDatasetLoader = baseDatasetLoader
        self.__instances = instancesToLoad

    def loadInterpolationData(self) -> List[InterpolateSubdataset]:
        _, _, (xTest, yTest) = self.__baseDatasetLoader.loadData()
        instancesTranspose = tuple(map(list, zip(*self.__instances)))
        xInstances = map(lambda instance: xTest[instance], instancesTranspose)
        yInstances = map(lambda instance: yTest[instance], instancesTranspose)
        return [InterpolateSubdataset("SPECIFIED_INSTANCES", *zip(xInstances, yInstances))]
