import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from experiments.Experiment import Experiment
from model.ConvolutionalAutoencoder import ConvolutionalAutoencoder
from datasets.NorbLoader import NorbLoader


from model.DenseAutoencoder import DenseAutoencoder
import numpy as np
import matplotlib.pyplot as plt


class TrainingNorbExperiment(Experiment):
    def run(self):
        # Hyperparameters
        originalImageDimensions = (96, 96)
        intermediateDimension = 256
        latentDimension = 10
        numConvolutions = 6
        baseConvolutionalDepth = 8

        # Build model and exhibit summary
        norbConvolutionalAutoencoder = ConvolutionalAutoencoder(
            originalImageDimensions,
            numConvolutions,
            baseConvolutionalDepth,
            intermediateDimension,
            latentDimension
        )
        # norbConvolutionalAutoencoder = DenseAutoencoder(
        #     originalImageDimensions,
        #     intermediateDimension,
        #     latentDimension
        # )
        norbConvolutionalAutoencoder.buildModels()
        norbConvolutionalAutoencoder.summary()


        # Obtain datasets and carry out normalisation
        norbLoader = NorbLoader('../res/norb')
        (xTrain, yTrain), (xTest, yTest) = norbLoader.loadData()
        xTrain = xTrain.astype('float32') / 255.
        xTest = xTest.astype('float32') / 255.

        # print (xTrain.shape, yTrain.shape, xTest.shape, yTest.shape)
        # xTrainMean = np.mean(xTrain, axis=0)
        # plt.figure(figsize=(1, 1))
        # num = plt.subplot(1, 1, 1)
        # plt.imshow(xTrainMean)
        # plt.gray()
        # num.get_xaxis().set_visible(False)
        # num.get_yaxis().set_visible(False)
        # plt.savefig('mean_norb.png')
        # quit()

        # Train model
        batchSize = 100
        epochs = 50
        norbConvolutionalAutoencoder.train(xTrain, xTest, epochs, batchSize)

        # Save network weights:
        norbConvolutionalAutoencoder.saveWeights("norbConvolutionalTrainingWeights.h5")
