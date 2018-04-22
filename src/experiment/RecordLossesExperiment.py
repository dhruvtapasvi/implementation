from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder
from results.ResultsStore import ResultsStore


class RecordLossesExperiment(Experiment):
    def __init__(self, datasplits, datasetName: str, variationalAutoencoder: VariationalAutoencoder, modelDescriptor: str, resultsStore: ResultsStore):
        self.__dataSplits = datasplits
        self.__datasetName = datasetName
        self.__variationalAutoencoder = variationalAutoencoder
        self.__modelDescriptor = modelDescriptor
        self.__resultsStore = resultsStore

    def run(self):
        (xTrain, _), (xVal, _), (xTest, _) = self.__dataSplits
        for split, splitName in [(xTrain, "train"), (xVal, "val"), (xTest, "test")]:
            splitLoss = self.__variationalAutoencoder.evaluate(split)
            self.__resultsStore.storeValue([self.__datasetName, self.__modelDescriptor, splitName], splitLoss)
