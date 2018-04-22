from experiment.Experiment import Experiment
from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
from model.Autoencoder import Autoencoder
from interpolate.Interpolate import Interpolate
from config.routes import getResultRouteStem

class InterpolateExperiment(Experiment):
    def __init__(self, interpolateData, autoencoder: Autoencoder, interpolate: Interpolate, resultRouteInner):
        self.__interpolateData = interpolateData
        self.__autoencoder = autoencoder
        self.__interpolate = interpolate
        self.__resultRouteStem = getResultRouteStem(resultRouteInner)

    def run(self):
        """
        TODO:
        Given left dataset, right dataset and autoencoder perform interpolation in latent space and in image space
        Print reconstructions as (Left, Interp, Right)
        Given controls CORRECT and COMPLETELY_INCORRECT (e.g. extrapolated beyond range), also construct above
        Then compare MET(Interp, CORRECT) and MET(Interp, COMPLETELY_INCORRECT)
        Variation: MET in { MSE-IS, MSE-LS, MBCE-IS)
        Variation: AUTOENC in { DENSE, CONV, PCA-DENSE, CVAE, IDENTITY }
        Variation: DATASET
        This experiment can loop over all metrics (first factor of variation
        The outer experiment can instantiate this one many times with the correct autoencoder
        (maybe a method to run with the autoencoder)
        What about reporting per class? Need to think about that in more details
        """
        interpolateDataset = self.__interpolateData.loadInterpolationData()
        for interpolateSubdataset in interpolateDataset:
            print(interpolateSubdataset.interpolatedFactorName)
            interpolated1 = self.__interpolate.interpolateAll(interpolateSubdataset.xLeft, interpolateSubdataset.xRight, 6)
            imagesArrayComparisonDisplay(interpolated1, self.__resultRouteStem + "interpolate_graduated_" + interpolateSubdataset.interpolatedFactorName + ".png")

            interpolated = self.__interpolate.interpolateAll(interpolateSubdataset.xLeft, interpolateSubdataset.xRight, 2)
            interpolateSubdatasetArrays = [
                interpolateSubdataset.xLeft,
                self.__autoencoder.autoencoder().predict_on_batch(interpolateSubdataset.xLeft),
                interpolateSubdataset.xRight,
                self.__autoencoder.autoencoder().predict_on_batch(interpolateSubdataset.xRight),
                interpolated[:, 2]
            ]
            if interpolateSubdataset.centreIsSpecified():
                interpolateSubdatasetArrays.append(interpolateSubdataset.xCentre)
                interpolateSubdatasetArrays.append(self.__autoencoder.autoencoder().predict_on_batch(interpolateSubdataset.xCentre))
                if interpolateSubdataset.outsideIsSpecified():
                    interpolateSubdatasetArrays.append(interpolateSubdataset.xOutside)
                    interpolateSubdatasetArrays.append(self.__autoencoder.autoencoder().predict_on_batch(interpolateSubdataset.xOutside))
            imagesArrayComparisonDisplay(interpolateSubdatasetArrays, self.__resultRouteStem + "interpolate_subdataset_" + interpolateSubdataset.interpolatedFactorName + ".png")
