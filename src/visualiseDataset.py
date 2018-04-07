import numpy as np

from config import routes

from dataset.loader.basic.ShapesBase import ShapesBase
from dataset.loader.basic.LoadFromFile import LoadFromFile

import dataset.info.ShapesInfo as shapesInfo

from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay


NUM_VISUALISATIONS = 100
datasetNamesAndLoaders = [
    ("shapesBase", ShapesBase()),
    ("shapesTransformed", LoadFromFile(routes.RESOURCE_ROUTE + shapesInfo.HOME, shapesInfo.BASE_IMAGE_SIZE, shapesInfo.BASE_IMAGE_SIZE))
]

for datasetName, datasetLoader in datasetNamesAndLoaders:
    (xTrain, _), (xVal, _), (xTest, _) = datasetLoader.loadData()
    if xTrain.shape[0] > NUM_VISUALISATIONS:
        np.random.shuffle(xTrain)
        xTrain = xTrain[:NUM_VISUALISATIONS]
    if xVal.shape[0] > NUM_VISUALISATIONS:
        np.random.shuffle(xVal)
        xVal = xVal[0:NUM_VISUALISATIONS]
    if xTest.shape[0] > NUM_VISUALISATIONS:
        np.random.shuffle(xTest)
        xTest = xTest[0:NUM_VISUALISATIONS]
    imagesArrayComparisonDisplay([xTrain, xVal, xTest], routes.getResultRouteStem("visualisations_") + datasetName)
