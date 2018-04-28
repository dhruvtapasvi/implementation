import scipy as sp
import numpy as np

from config import routes
from config.VaeConfig import VaeConfig
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder


class SamplingExperiment(Experiment):
    def __init__(self, dataSplits, config: VaeConfig, variationalAutoencoder: VariationalAutoencoder, numSamples, resultRouteInner: str):
        self.__dataSplits = dataSplits
        self.__resultRouteStem = routes.getResultRouteStem(resultRouteInner)
        self.__latentDimension = config.latentRepresentationDimension
        self.__decoder = variationalAutoencoder.decoder()
        self.__numSamples = numSamples

    def run(self):
        randomSamples = sp.randn(self.__numSamples, self.__latentDimension)
        decodedRandomSamples = np.array([self.__decoder.predict(randomSamples, batch_size=100)]).swapaxes(0, 1)
        imagesArrayComparisonDisplay(decodedRandomSamples, self.__resultRouteStem + "randomSampling.png")
