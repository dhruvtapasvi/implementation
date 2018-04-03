import pickle

from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder
from config.VaeConfig import VaeConfig


class SaveModelTrainingExperiment(Experiment):
    def __init__(self, model: VariationalAutoencoder, variationalAutoencoderConfig: VaeConfig, modelTrainingHistory):
        self.__model = model
        self.__variationalAutoencoderConfig = variationalAutoencoderConfig
        self.__modelTrainingHistory = modelTrainingHistory

    def run(self):
        """
        Save the model and the associated training history in the appropriate location
        :return: Nothing
        """
        rootPath = ".."
        modelWeightsPath = rootPath + "/cacheWeights"
        modelTrainingHistoryPath = rootPath + "/modelTrainingHistory"

        pickle.dump(
            modelTrainingHistoryPath,
            open(modelTrainingHistoryPath + "/" + self.__variationalAutoencoderConfig.stringDescriptor + ".history.p", "wb")
        )
        self.__model.saveWeights(modelWeightsPath + "/" + ".weights.h5")
