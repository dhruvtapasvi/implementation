from dataset.interpolate import InterpolateDatasetLoader
from dataset.loader import DatasetLoader


class DatasetPackage:
    def __init__(self, name: str, datasetLoader: DatasetLoader, interpolateLoader: InterpolateDatasetLoader):
        self.__name = name
        self.__datasetLoader = datasetLoader
        self.__interpolateLoader = interpolateLoader

    @property
    def name(self):
        return self.__name

    @property
    def datasetLoader(self):
        return self.__datasetLoader

    @property
    def interpolateLoader(self):
        return self.__interpolateLoader
