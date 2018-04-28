from dataset.DatasetPackage import DatasetPackage
from config.VaeConfig import VaeConfig
from evaluation.metric.Metric import Metric


class ExperimentalConfigTuple:
    def __init__(self, datasetPackage: DatasetPackage, config: VaeConfig, batchSize: int, epochs: int, metricLatentSpace: Metric, metricImageSpace: Metric):
        self.__datasetPackage = datasetPackage
        self.__config = config
        self.__batchSize = batchSize
        self.__epochs = epochs
        self.__metricLatentSpace = metricLatentSpace
        self.__metricImageSpace = metricImageSpace

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

    @property
    def metricLatentSpace(self):
        return self.__metricLatentSpace

    @property
    def metricImageSpace(self):
        return self.__metricImageSpace
