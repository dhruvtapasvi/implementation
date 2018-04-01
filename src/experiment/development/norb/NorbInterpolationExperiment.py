import numpy as np

from dataset.info.NorbInfo import NorbLabelIndex
from dataset.loader.basic.NorbLoader import NorbLoader
from dataset.loader.preprocess import ScaleBetweenZeroAndOne
from dataset.loader.preprocess import SortByLabels
from dataset.process.FilterDatasetLabelPredicate import  FilterDatasetLabelPredicate
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
from experiment.Experiment import Experiment
from interpolate.Interpolate import Interpolate
from metric.SquaredError import SquaredError


class NorbInterpolationExperiment(Experiment):
    def __init__(self):
        print("Init")

    def run(self):
        norbLoader = SortByLabels(ScaleBetweenZeroAndOne(NorbLoader("./res/norb"), 0, 255))
        _, _, (xTest, yTest) = norbLoader.loadData()
        filter = FilterDatasetLabelPredicate()
        xLowElev, _ = filter.filter(xTest, yTest, lambda y: y[NorbLabelIndex.ELEVATION.value] == 0)
        xMediumElev, _ = filter.filter(xTest, yTest, lambda y: y[NorbLabelIndex.ELEVATION.value] == 4)
        xHighElev, _ = filter.filter(xTest, yTest, lambda y: y[NorbLabelIndex.ELEVATION.value] == 8)
        interpolate = Interpolate()
        xInterp = interpolate.interpolateAll(xLowElev, xHighElev, 2)[:, 1]
        print(xInterp.shape)
        arraysToDisplay = [xLowElev, xMediumElev, xInterp, xHighElev]
        numDisplay = 20
        imagesArrayComparisonDisplay(arraysToDisplay, "./out/interexamplenorb", endIndex=numDisplay)
        squaredError = SquaredError(variance=1)
        print(np.mean(squaredError.compute(xLowElev, xInterp)))
        print(np.mean(squaredError.compute(xMediumElev, xInterp)))
        print(np.mean(squaredError.compute(xHighElev, xInterp)))
