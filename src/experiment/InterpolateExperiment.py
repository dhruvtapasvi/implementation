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
            resultsStore: ResultsStore,
            numSampleInterpolations,
            numIntervals):
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
        self.__numSampleInterpolations = numSampleInterpolations
        self.__numIntervals = numIntervals

    def run(self):
        for interpolationSubdataset in self.__interpolateData:
            self.__numericalInterpolationMetrics(interpolationSubdataset)
            self.__visualInterpolation(interpolationSubdataset)

        if len(self.__interpolateData) > 1:
            combineSubdatasets = CombineInterpolateLoaders()
            combinedSubdatasets = combineSubdatasets.combine(self.__interpolateData, "COMBINED")
            self.__numericalInterpolationMetrics(combinedSubdatasets)
            self.__visualInterpolation(combinedSubdatasets)

    def __numericalInterpolationMetrics(self, interpolateSubdataset: InterpolateSubdataset):
        if interpolateSubdataset.centreIsSpecified():
            interpolated, interpolatedReconstructed = self.__interpolateLatentSpace.interpolateAll(
                interpolateSubdataset.xLeft, interpolateSubdataset.xRight, 2)
            interpolatedReconstructed = interpolatedReconstructed[:, 1]
            interpolated = interpolated[:, 1]

            interpolatedImageSpace = self.__interpolate.interpolateAll(interpolateSubdataset.xLeft,
                                                                       interpolateSubdataset.xRight, 2)[:, 1]

            randomImages = np.random.random_sample(interpolateSubdataset.xLeft.shape)

            xCentreEncoded = self.__autoencoder.encoder().predict(interpolateSubdataset.xCentre, batch_size=100)

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "interpolateLatentSpace", "metricImageSpace"],
                self.__imageSpaceComparisonMetric.compute(interpolateSubdataset.xCentre, interpolatedReconstructed)
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "interpolateLatentSpace", "metricLatentSpace"],
                self.__latentSpaceComparisonMetric.compute(xCentreEncoded, interpolated)
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "interpolateImageSpace", "metricImageSpace"],
                self.__imageSpaceComparisonMetric.compute(interpolateSubdataset.xCentre, interpolatedImageSpace)
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "interpolateImageSpace", "metricLatentSpace"],
                self.__latentSpaceComparisonMetric.compute(
                    xCentreEncoded,
                    self.__autoencoder.encoder().predict(interpolatedImageSpace, batch_size=100),
                )
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "randomImage", "metricImageSpace"],
                self.__imageSpaceComparisonMetric.compute(interpolateSubdataset.xCentre, randomImages)
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "randomImage", "metricLatentSpace"],
                self.__latentSpaceComparisonMetric.compute(
                    xCentreEncoded,
                    self.__autoencoder.encoder().predict(randomImages, batch_size=100),
                )
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "left", "metricImageSpace"],
                self.__imageSpaceComparisonMetric.compute(interpolateSubdataset.xCentre, interpolateSubdataset.xLeft)
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "left", "metricLatentSpace"],
                self.__latentSpaceComparisonMetric.compute(
                    xCentreEncoded,
                    self.__autoencoder.encoder().predict(interpolateSubdataset.xLeft, batch_size=100),
                )
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "right", "metricImageSpace"],
                self.__imageSpaceComparisonMetric.compute(interpolateSubdataset.xCentre, interpolateSubdataset.xRight)
            )

            self.__resultsStore.storeValue(
                [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "right", "metricLatentSpace"],
                self.__latentSpaceComparisonMetric.compute(
                    xCentreEncoded,
                    self.__autoencoder.encoder().predict(interpolateSubdataset.xRight, batch_size=100),
                )
            )

            if interpolateSubdataset.outsideIsSpecified():
                self.__resultsStore.storeValue(
                    [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "outside", "metricImageSpace"],
                    self.__imageSpaceComparisonMetric.compute(interpolateSubdataset.xCentre, interpolateSubdataset.xOutside)
                )

                self.__resultsStore.storeValue(
                    [self.__datasetName, self.__modelName, interpolateSubdataset.interpolatedFactorName, "outside", "metricLatentSpace"],
                    self.__latentSpaceComparisonMetric.compute(
                        xCentreEncoded,
                        self.__autoencoder.encoder().predict(interpolateSubdataset.xOutside, batch_size=100)
                    )
                )

    def __visualInterpolation(self, interpolateSubdataset: InterpolateSubdataset):
        randomSubset = np.random.choice(len(interpolateSubdataset.xLeft), min(self.__numSampleInterpolations, len(interpolateSubdataset.xLeft)), replace=False)
        truncatedInterpolateSubdataset = InterpolateSubdataset(
            interpolateSubdataset.interpolatedFactorName,
            (interpolateSubdataset.xLeft[randomSubset], interpolateSubdataset.yLeft[randomSubset]),
            (interpolateSubdataset.xRight[randomSubset], interpolateSubdataset.yRight[randomSubset]),
            (interpolateSubdataset.xCentre[randomSubset], interpolateSubdataset.yCentre[randomSubset])
                if interpolateSubdataset.centreIsSpecified() else None,
            (interpolateSubdataset.xOutside[randomSubset], interpolateSubdataset.yOutside[randomSubset])
                if interpolateSubdataset.centreIsSpecified() else None
        )
        _, interpolatedReconstructed = self.__interpolateLatentSpace.interpolateAll(truncatedInterpolateSubdataset.xLeft, truncatedInterpolateSubdataset.xRight, self.__numIntervals)
        interpolatedReconstructed = np.swapaxes(interpolatedReconstructed, 0, 1)
        interpolatedDisplay = np.concatenate([np.array([truncatedInterpolateSubdataset.xLeft]), interpolatedReconstructed, np.array([truncatedInterpolateSubdataset.xRight])])
        imagesArrayComparisonDisplay(interpolatedDisplay, self.__resultRouteStem + interpolateSubdataset.interpolatedFactorName + "_interpolated_graduated.png")
