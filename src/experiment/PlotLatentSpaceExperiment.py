import numpy as np
from scipy.stats import norm


from config import routes
from config.VaeConfig import VaeConfig
from dataset.loader.DatasetLoader import DatasetLoader
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
from display.scatterPlotDisplay import scatterPlotDisplay
from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder


class ReconstructionsExperiment(Experiment):
    def __init__(self, datasetModelPath:str, datasetLoader: DatasetLoader, config: VaeConfig, variationalAutoencoder: VariationalAutoencoder, sqrtNumSamples, labelToValue):
        self.__datasetLoader = datasetLoader
        self.__resultRouteStem = routes.getResultRouteStem(datasetModelPath)
        self.__latentDimension = config.latentRepresentationDimension
        self.__encoder = variationalAutoencoder.encoder()
        self.__decoder = variationalAutoencoder.decoder()
        self.__sqrtNumSamples = sqrtNumSamples
        self.__labelToValue = labelToValue

    def run(self):
        """
        Plot points in the latent space
        """
        assert(self.__latentDimension == 2)

        _, _, (xTest, yTest) = self.__datasetLoader.loadData()

        # Draw where each test sample lies in the latent space
        xTestEncoded = self.__encoder.predict(xTest, batch_size=100)
        values = np.array([self.__labelToValue(label) for label in yTest])
        scatterPlotRoute = self.__resultRouteStem + "LatentSpacePlot.png"
        scatterPlotDisplay(xTestEncoded, values, scatterPlotRoute)
