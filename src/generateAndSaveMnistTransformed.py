import os
import numpy as np


import dataset.info.MnistTransformedInfo as mnistTransformedInfo
from dataset.loader.basic.MnistLoader import MnistLoader
from dataset.loader.preprocess.RandomTransforms import RandomTransforms
import config.routes as routes


resourceRoot = routes.RESOURCE_ROUTE
if not os.path.isdir(resourceRoot):
    os.mkdir(resourceRoot)
datasetRouteStem = resourceRoot + "/mnistTransformed_"

mnistLoader = MnistLoader()
numRandomTransforms = [1, 2, 5, 10]

for i in numRandomTransforms:
    print("Creating MNIST transformed dataset with " + str(i) + " random transformations per training example...")

    datasetRoute = datasetRouteStem + str(i)
    if not os.path.isdir(datasetRoute):
        os.mkdir(datasetRoute)

    randomLoader = RandomTransforms(
        mnistLoader,
        mnistTransformedInfo.TRANSFORM_SHEAR_FACTOR,
        mnistTransformedInfo.TRANSFORM_LOG2_STRETCH_FACTOR,
        i,
        randomSeed=mnistTransformedInfo.RANDOM_GENERATION_SEED
    )
    (xTrain, yTrain), (xVal, yVal), (xTest, yTest) = randomLoader.loadData()

    np.save(datasetRoute + "/x_train.npy", xTrain)
    np.save(datasetRoute + "/y_train.npy", yTrain)
    np.save(datasetRoute + "/x_val.npy", xVal)
    np.save(datasetRoute + "/y_val.npy", yVal)
    np.save(datasetRoute + "/x_test.npy", xTest)
    np.save(datasetRoute + "/y_test.npy", yTest)

    print("Created MNIST transformed dataset with " + str(i) + " random transformations per training example!")
