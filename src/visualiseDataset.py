import numpy as np
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
import dataset.loaderPackaged as loaders
from config import routes


NUM_VISUALISATIONS = 100
datasetPackages = [
    loaders.shapesPackage,
    loaders.shapesTransformedPackage
]

for datasetPackage in datasetPackages:
    (xTrain, _), (xVal, _), (xTest, _) = datasetPackage.datasetLoader.loadData()
    if xTrain.shape[0] > NUM_VISUALISATIONS:
        np.random.shuffle(xTrain)
        xTrain = xTrain[:NUM_VISUALISATIONS]
    if xVal.shape[0] > NUM_VISUALISATIONS:
        np.random.shuffle(xVal)
        xVal = xVal[0:NUM_VISUALISATIONS]
    if xTest.shape[0] > NUM_VISUALISATIONS:
        np.random.shuffle(xTest)
        xTest = xTest[0:NUM_VISUALISATIONS]
    imagesArrayComparisonDisplay([xTrain, xVal, xTest], routes.getResultRouteStem("visualisations_") + datasetPackage.name)
