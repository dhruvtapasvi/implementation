import math
import numpy as np
import os

from dataset.basicLoader.MnistLoader import MnistLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from dataset.preprocessLoader.RandomTransforms import RandomTransforms
from experiment.Experiment import Experiment
import dataset.info.MnistTransformedInfo as mnistTransformedInfo


class TransformMnistExperiment(Experiment):
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
                randomLoader = RandomTransforms(mnistLoader, 0.14, 1 / 2, i, mnistTransformedInfo.RANDOM_GENERATION_SEED)
                (xTrain, yTrain), (xVal, yVal), (xTest, yTest) = randomLoader.loadData()
                print(xTrain.shape)

                np.save(folderPath + "/x_train", xTrain)
                np.save(folderPath + "/y_train", yTrain)
                np.save(folderPath + "/x_val", xVal)
                np.save(folderPath + "/y_val", yVal)
                np.save(folderPath + "/x_test", xTest)
                np.save(folderPath + "/y_test", yTest)

            print("Tranformed " + str(i) + " dataset created")
