import os

import numpy as np

import dataset.info.MnistTransformedInfo as mnistTransformedInfo
from dataset.loader.basic.MnistLoader import MnistLoader
from dataset.loader.preprocess.RandomTransforms import RandomTransforms
from experiment.Experiment import Experiment


class CreateMnistTransformedExperiment(Experiment):
    def run(self):
        mnistLoader = MnistLoader()
        datasetRoute = "../res"
        if not os.path.isdir(datasetRoute):
            os.mkdir(datasetRoute)
        for i in [1, 2, 5, 10]:
            print("Creating " + str(i) + " transformed dataset...")
            folderPath = datasetRoute + "/mnistTransformed_" + str(i)
            if not os.path.isdir(folderPath):
                os.mkdir(folderPath)
                randomLoader = RandomTransforms(
                    mnistLoader,
                    mnistTransformedInfo.TRANSFORM_SHEAR_FACTOR,
                    mnistTransformedInfo.TRANSFORM_LOG2_STRETCH_FACTOR,
                    i,
                    mnistTransformedInfo.RANDOM_GENERATION_SEED
                )
                (xTrain, yTrain), (xVal, yVal), (xTest, yTest) = randomLoader.loadData()
                print(xTrain.shape)

                np.save(folderPath + "/x_train", xTrain)
                np.save(folderPath + "/y_train", yTrain)
                np.save(folderPath + "/x_val", xVal)
                np.save(folderPath + "/y_val", yVal)
                np.save(folderPath + "/x_test", xTest)
                np.save(folderPath + "/y_test", yTest)

            print("Tranformed " + str(i) + " dataset created")
