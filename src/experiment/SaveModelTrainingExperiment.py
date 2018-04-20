import pickle

from config import routes
from config.VaeConfig import VaeConfig
from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder


class SaveModelTrainingExperiment(Experiment):
    def __init__(self, model: VariationalAutoencoder, modelTrainingHistory, savePathStem: str):
        self.__model = model
        self.__modelTrainingHistory = modelTrainingHistory
        self.__savePathStem = savePathStem

    def run(self):
        """
        Save the model and the associated training history in the appropriate location
        """
        self.__model.saveWeights(routes.getModelWeightsRoute(self.__savePathStem))

        pickle.dump(
            self.__modelTrainingHistory,
            open(routes.getModelTrainingHistoryRoute(self.__savePathStem), "wb")
        )
