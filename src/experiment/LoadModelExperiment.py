from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder
from config.VaeConfig import VaeConfig
from config import routes


class LoadModelExperiment(Experiment):
    def __init__(self, weightsPathInner: str, variationalAutoencoder: VariationalAutoencoder):
        self.__weightsPath = routes.getModelWeightsRoute(weightsPathInner)
        self.__variationalAutoencoder = variationalAutoencoder

    def run(self):
        """
        Load the weights into a specified model
        """
        self.__variationalAutoencoder.loadWeights(self.__weightsPath)
