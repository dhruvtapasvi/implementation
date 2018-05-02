import numpy as np
from display.imagesArraysComparisonDisplay import imagesArrayComparisonDisplay
import dataset.loaderPackaged as loaders
from config import routes
from dataset.interpolate.process.CombineInterpolateLoaders import CombineInterpolateLoaders
from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset


NUM_VISUALISATIONS = 10
datasetPackages = [
    loaders.mnistPackage,
    loaders.shapesTransformedLimitedRotationPackage,
    loaders.mnistTransformedLimitedRotationPackage,
    loaders.norbPackage,
    # loaders.shapesPackage
]

# for datasetPackage in datasetPackages:
#     (xTrain, _), (xVal, _), (xTest, _) = datasetPackage.datasetLoader.loadData()
#     if xTrain.shape[0] > NUM_VISUALISATIONS:
#         np.random.shuffle(xTrain)
#         xTrain = xTrain[:NUM_VISUALISATIONS]
#     if xVal.shape[0] > NUM_VISUALISATIONS:
#         np.random.shuffle(xVal)
#         xVal = xVal[0:NUM_VISUALISATIONS]
#     if xTest.shape[0] > NUM_VISUALISATIONS:
#         np.random.shuffle(xTest)
#         xTest = xTest[0:NUM_VISUALISATIONS]
#     imagesArrayComparisonDisplay(np.array([xTrain, xVal, xTest]).swapaxes(0, 1), routes.getResultRouteStem("visualisations_") + datasetPackage.name)
#

combine = CombineInterpolateLoaders()


NUM_INTERP_VISUALISATIONS = 8

for datasetPackage in datasetPackages:
    interpolateSubdataset = combine.combine(datasetPackage.interpolateLoader.loadInterpolationData())
    randomSubset = np.random.choice(len(interpolateSubdataset.xLeft),
                                    min(NUM_INTERP_VISUALISATIONS, len(interpolateSubdataset.xLeft)),
                                    replace=False)
    truncatedInterpolateSubdataset = InterpolateSubdataset(
        interpolateSubdataset.interpolatedFactorName,
        (interpolateSubdataset.xLeft[randomSubset], interpolateSubdataset.yLeft[randomSubset]),
        (interpolateSubdataset.xRight[randomSubset], interpolateSubdataset.yRight[randomSubset]),
        (interpolateSubdataset.xCentre[randomSubset], interpolateSubdataset.yCentre[randomSubset])
        if interpolateSubdataset.centreIsSpecified() else None,
        (interpolateSubdataset.xOutside[randomSubset], interpolateSubdataset.yOutside[randomSubset])
        if interpolateSubdataset.centreIsSpecified() else None
    )

    arrays = ([truncatedInterpolateSubdataset.xCentre] if truncatedInterpolateSubdataset.centreIsSpecified() else []) + \
        [truncatedInterpolateSubdataset.xLeft, truncatedInterpolateSubdataset.xRight] + \
        ([truncatedInterpolateSubdataset.xOutside] if truncatedInterpolateSubdataset.outsideIsSpecified() else [])

    print(list(map(len, arrays)))

    imagesArrayComparisonDisplay(np.array(arrays).swapaxes(0, 1), routes.getResultRouteStem("interpolationVisualisations") + datasetPackage.name)
