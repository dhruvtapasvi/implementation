from experiment.Experiment import Experiment
from config.VaeConfig import VaeConfig


class BuildModelExperiment(Experiment):
    def __init__(self, variationalAutoencoderConfig: VaeConfig):
        self.__variationalAutoencoderConfig = variationalAutoencoderConfig

    def run(self):
        """
        Create a variational autoencoder from the specified configuration instance, and publish a summary.
        :return: Variational autoencoder instance
        """
        variationalAutoencoder = self.__variationalAutoencoderConfig.fromConfig()
        variationalAutoencoder.buildModels()
        variationalAutoencoder.summary()
        return variationalAutoencoder
