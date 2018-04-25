import numpy as np

from experiment.Experiment import Experiment
from dataset.interpolate.process.CombineInterpolateLoaders import CombineInterpolateLoaders
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
from model.Autoencoder import Autoencoder
from interpolate.Interpolate import Interpolate
from interpolate.InterpolateLatentSpace import InterpolateLatentSpace
from config.routes import getResultRouteStem
from evaluation.metric.Metric import Metric
from evaluation.results.ResultsStore import ResultsStore
from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset

class InterpolateExperiment(Experiment):
    def __init__(
            self,
            interpolateData,
            datasetName: str,
            autoencoder: Autoencoder,
            modelName: str,
            latentSpaceComparisonMetric: Metric,
            imageSpaceComparisonMetric: Metric,
            resultRouteInner: str,
            resultsStore: ResultsStore):
        self.__interpolateData = interpolateData
        self.__datasetName = datasetName
        self.__autoencoder = autoencoder
        self.__modelName = modelName
        self.__latentSpaceComparisonMetric = latentSpaceComparisonMetric
        self.__imageSpaceComparisonMetric = imageSpaceComparisonMetric
        self.__resultRouteStem = getResultRouteStem(resultRouteInner)
        self.__resultsStore = resultsStore
        self.__interpolate = Interpolate()
        self.__interpolateLatentSpace = InterpolateLatentSpace(autoencoder)

    def run(self):
        for interpolationSubdataset in self.__interpolateData:
            self.__numericalInterpolationMetrics(interpolationSubdataset)
            self.__visualInterpolation(interpolationSubdataset)

        if len(self.__interpolateData) > 1:
            combineSubdatasets = CombineInterpolateLoaders()
            combinedSubdatasets = combineSubdatasets.combine(self.__interpolateData, "Combined")
            self.__numericalInterpolationMetrics(combinedSubdatasets)

    def __numericalInterpolationMetrics(self, interpolateSubdataset: InterpolateSubdataset):
        interpolated, interpolatedReconstructed = self.__interpolateLatentSpace.interpolateAll(interpolateSubdataset.xLeft, interpolateSubdataset.xRight, 2)

        interpolatedReconstructed = interpolatedReconstructed[:, 1]
        interpolated = interpolated[:, 1]

        if interpolateSubdataset.centreIsSpecified():
            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "actual", "metricImageSpace"],
                self.__imageSpaceComparisonMetric.compute(interpolateSubdataset.xCentre, interpolatedReconstructed).mean
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "actual", "metricLatentSpace"],
                self.__latentSpaceComparisonMetric.compute(
                    self.__autoencoder.encoder().predict(interpolateSubdataset.xCentre, batch_size=100),
                    interpolated
                ).mean
            )

            if interpolateSubdataset.outsideIsSpecified():
                self.__resultsStore.storeValue(
                    [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "control", "metricImageSpace"],
                    self.__imageSpaceComparisonMetric.compute(interpolateSubdataset.xCentre, interpolateSubdataset.xOutside).mean
                )

                self.__resultsStore.storeValue(
                    [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "control", "metricLatentSpace"],
                    self.__latentSpaceComparisonMetric.compute(
                        self.__autoencoder.encoder().predict(interpolateSubdataset.xCentre, batch_size=100),
                        self.__autoencoder.encoder().predict(interpolateSubdataset.xOutside, batch_size=100)
                    ).mean
                )

    def __visualInterpolation(self, interpolateSubdataset: InterpolateSubdataset):
        _, interpolatedReconstructed = self.__interpolateLatentSpace.interpolateAll(interpolateSubdataset.xLeft, interpolateSubdataset.xRight, 6)
        interpolatedReconstructed = np.swapaxes(interpolatedReconstructed, 0, 1)
        interpolatedDisplay = np.concatenate([np.array([interpolateSubdataset.xLeft]), interpolatedReconstructed, np.array([interpolateSubdataset.xRight])])
        imagesArrayComparisonDisplay(interpolatedDisplay, self.__resultRouteStem + interpolateSubdataset.interpolatedFactorName + "_interpolated_graduated.png")
