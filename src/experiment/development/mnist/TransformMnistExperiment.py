import math
import numpy as np
import os

from dataset.basicLoader.MnistLoader import MnistLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from dataset.preprocessLoader.RandomTransforms import RandomTransforms
from experiment.Experiment import Experiment


class TransformMnistExperiment(Experiment):
    def run(self):
        mnistLoader = ScaleBetweenZeroAndOne(MnistLoader(), 0, 255)
        datasetRoute = "../res"
        if not os.path.isdir(datasetRoute):
            os.mkdir(datasetRoute)
        for i in [1, 2, 5, 10]:
            print("Creating " + str(i) + " transformed dataset...")
            folderPath = datasetRoute + "/mnistTransformed_" + str(i)
            if not os.path.isdir(folderPath):
                os.mkdir(folderPath)
                randomLoader = RandomTransforms(mnistLoader, 0.14, 1 / 2, i)
                (xTrain, yTrain), (xVal, yVal), (xTest, yTest) = randomLoader.loadData()
                print(xTrain.shape)

                np.save(datasetRoute + "/x_train", xTrain)
                np.save(datasetRoute + "/y_train", yTrain)
                np.save(datasetRoute + "/x_val", xVal)
                np.save(datasetRoute + "/y_val", yVal)
                np.save(datasetRoute + "/x_test", xTest)
                np.save(datasetRoute + "/y_test", yTest)

            print("Tranformed " + str(i) + " dataset created")
