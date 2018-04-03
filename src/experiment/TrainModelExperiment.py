from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder
from dataset.loader.DatasetLoader import DatasetLoader


class TrainModelExperiment(Experiment):
    def __init__(self, variationalAutoencoder: VariationalAutoencoder, datasetLoader: DatasetLoader, epochs, batchSize):
        self.__variationalAutoencoder = variationalAutoencoder
        self.__datasetLoader = datasetLoader
        self.__epochs = epochs
        self.__batchSize = batchSize

    def run(self):
        """
        Train the model specified in the constructor with the parameters specified there too
        Side effect: the model is trained
        :return: The model training history
        """
        (xTrain, _), (xValidation, _), _ = self.__datasetLoader.loadData()
        modelTrainingHistory = self.__variationalAutoencoder.train(xTrain, xValidation, self.__epochs, self.__batchSize)
        return modelTrainingHistory.history
