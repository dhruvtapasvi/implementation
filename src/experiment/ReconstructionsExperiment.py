import scipy as sp
import numpy as np

from config import routes
from config.VaeConfig import VaeConfig
from dataset.loader.DatasetLoader import DatasetLoader
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder


class ReconstructionsExperiment(Experiment):
    def __init__(self, datasetLoader: DatasetLoader, config: VaeConfig, variationalAutoencoder: VariationalAutoencoder, numSampleReconstructions, sqrtNumSamples):
        self.__datasetLoader = datasetLoader
        self.__resultRouteStem = routes.getResultRouteStem(config.stringDescriptor)
        self.__latentDimension = config.latentRepresentationDimension
        self.__autoencoder = variationalAutoencoder.autoencoder()
        self.__decoder = variationalAutoencoder.decoder()
        self.__numSampleReconstructions = numSampleReconstructions
        self.__sqrtNumSamples = sqrtNumSamples

    def run(self):
        """
        Create reconstructions and random sampling from the variational autoencoder
        """
        (xTrain, _), (xVal, _), (xTest, _) = self.__datasetLoader.loadData()
        np.random.shuffle(xTrain)
        np.random.shuffle(xVal)
        np.random.shuffle(xTest)

        self.__displayReconstructions(xTrain, "train")
        self.__displayReconstructions(xVal, "validation")
        self.__displayReconstructions(xTest, "test")

        randomSamples = sp.randn(self.__sqrtNumSamples * self.__sqrtNumSamples, self.__latentDimension)
        decodedRandomSamples = self.__decoder.predict_on_batch(randomSamples)
        imagesArrayComparisonDisplay(
            decodedRandomSamples.reshape((self.__sqrtNumSamples, self.__sqrtNumSamples) + decodedRandomSamples.shape[1:]),
            self.__resultRouteStem + "randomSampling.png"
        )

    def __displayReconstructions(self, xData, reconstructionType: str):
        xDataTruncated = xData[:self.__numSampleReconstructions]
        xDataTruncatedReconstructed = self.__autoencoder.predict_on_batch(xDataTruncated)
        imagesArrayComparisonDisplay([xDataTruncated, xDataTruncatedReconstructed], self.__resultRouteStem + reconstructionType + "Reconstructions.png")
