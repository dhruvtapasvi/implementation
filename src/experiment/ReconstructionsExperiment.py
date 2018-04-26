import scipy as sp
import numpy as np

from config import routes
from config.VaeConfig import VaeConfig
from dataset.loader.DatasetLoader import DatasetLoader
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder


class ReconstructionsExperiment(Experiment):
    def __init__(self, dataSplits, variationalAutoencoder: VariationalAutoencoder, numSampleReconstructions, resultRouteInner: str):
        self.__dataSplits = dataSplits
        self.__resultRouteStem = routes.getResultRouteStem(resultRouteInner)
        self.__autoencoder = variationalAutoencoder.autoencoder()
        self.__numSampleReconstructions = numSampleReconstructions

    def run(self):
        """
        Create reconstructions and random sampling from the variational autoencoder
        """
        (xTrain, _), (xVal, _), (xTest, _) = self.__dataSplits
        xTrainRandomTrunc = xTrain[np.random.choice(len(xTrain), self.__numSampleReconstructions, replace=False)]
        xTestRandomTrunc = xTest[np.random.choice(len(xTest), self.__numSampleReconstructions, replace=False)]

        self.__displayReconstructions(xTrainRandomTrunc, "train")
        self.__displayReconstructions(xTestRandomTrunc, "test")

    def __displayReconstructions(self, xData, reconstructionType: str):
        xDataTruncated = xData[:self.__numSampleReconstructions]
        xDataTruncatedReconstructed = self.__autoencoder.predict(xDataTruncated, batch_size=100)
        arraysToPrint = np.array([xDataTruncated, xDataTruncatedReconstructed]).swapaxes(0, 1)
        imagesArrayComparisonDisplay(arraysToPrint, self.__resultRouteStem + reconstructionType + "Reconstructions.png")
