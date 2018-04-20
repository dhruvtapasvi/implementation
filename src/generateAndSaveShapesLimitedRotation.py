import os
import numpy as np


import dataset.info.ShapesInfoLimitedRotation as shapesInfoLimitedRotations
from dataset.loader.basic.ShapesBase import ShapesBase
from dataset.loader.preprocess.RandomTransforms import RandomTransforms
import config.routes as routes


resourceRoot = routes.RESOURCE_ROUTE
if not os.path.isdir(resourceRoot):
    os.mkdir(resourceRoot)
datasetRoute = resourceRoot + shapesInfoLimitedRotations.HOME
if not os.path.isdir(datasetRoute):
    os.mkdir(datasetRoute)

randomLoader = RandomTransforms(
    ShapesBase(),
    shapesInfoLimitedRotations.TRANSFORM_MIN_MAX_ROTATIONS,
    shapesInfoLimitedRotations.TRANSFORM_SHEAR_FACTOR,
    shapesInfoLimitedRotations.TRANSFORM_LOG2_STRETCH_FACTOR,
    numSamplesPerTrainPoint=shapesInfoLimitedRotations.BASE_TRAIN_PROPORTION * shapesInfoLimitedRotations.BASE_NUM_SAMPLES,
    numSamplesPerValidationPoint=shapesInfoLimitedRotations.BASE_VALIDATION_PROPORTION * shapesInfoLimitedRotations.BASE_NUM_SAMPLES,
    numSamplesPerTestPoint=shapesInfoLimitedRotations.BASE_TEST_PROPORTION * shapesInfoLimitedRotations.BASE_NUM_SAMPLES,
    randomSeed=shapesInfoLimitedRotations.RANDOM_GENERATION_SEED
)
(xTrain, yTrain), (xVal, yVal), (xTest, yTest) = randomLoader.loadData()
np.save(datasetRoute + "/x_train.npy", xTrain)
np.save(datasetRoute + "/y_train.npy", yTrain)
np.save(datasetRoute + "/x_val.npy", xVal)
np.save(datasetRoute + "/y_val.npy", yVal)
np.save(datasetRoute + "/x_test.npy", xTest)
np.save(datasetRoute + "/y_test.npy", yTest)
