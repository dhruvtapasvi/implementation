import pickle

from config import routes
from config.VaeConfig import VaeConfig
from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder


class SaveModelTrainingExperiment(Experiment):
    def __init__(self, model: VariationalAutoencoder, variationalAutoencoderConfig: VaeConfig, modelTrainingHistory):
        self.__model = model
        self.__variationalAutoencoderConfig = variationalAutoencoderConfig
        self.__modelTrainingHistory = modelTrainingHistory

    def run(self):
        """
        Save the model and the associated training history in the appropriate location
        """
        self.__model.saveWeights(routes.getModelWeightsRoute(self.__variationalAutoencoderConfig.stringDescriptor))

        pickle.dump(
            self.__modelTrainingHistory,
            open(routes.getModelTrainingHistoryRoute(self.__variationalAutoencoderConfig.stringDescriptor), "wb")
        )
