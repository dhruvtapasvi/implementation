from dataset.DatasetPackage import DatasetPackage
from config.VaeConfig import VaeConfig


class ExperimentalConfigTuple:
    def __init__(self, datasetPackage: DatasetPackage, config: VaeConfig, batchSize: int, epochs: int):
        self.__datasetPackage = datasetPackage
        self.__config = config
        self.__batchSize = batchSize
        self.__epochs = epochs

    @property
    def datasetPackage(self):
        return self.__datasetPackage

    @property
    def config(self):
        return self.__config

    @property
    def batchSize(self):
        return self.__batchSize

    @property
    def epochs(self):
        return self.__epochs

    @property
    def stringDescriptor(self):
        return self.__datasetPackage.name + "_" + self.__config.stringDescriptor
