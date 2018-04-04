from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder
from config.VaeConfig import VaeConfig
from config import routes


class LoadModelExperiment(Experiment):
    def __init__(self, variationalAutoencoderConfig: VaeConfig, variationalAutoencoder: VariationalAutoencoder):
        self.__variationalAutoencoderConfig = variationalAutoencoderConfig
        self.__variationalAutoencoder = variationalAutoencoder

    def run(self):
        """
        Load the weights into a specified model
        """
        weightsPath = routes.getModelWeightsRoute(self.__variationalAutoencoderConfig.stringDescriptor)
        print("Loading model " + weightsPath)
        self.__variationalAutoencoder.loadWeights(weightsPath)
