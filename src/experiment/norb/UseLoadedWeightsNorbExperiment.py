import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from experiment.Experiment import Experiment
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne


class UseLoadedWeightsNorbExperiment(Experiment):
    def run(self):
        config = ConvolutionalAutoencoderConfig("./config/model/convolutional/norb_conv_6_16_256_10_bce.json")

        # Build model and exhibit summary
        norbAutoencoder = config.fromConfig()
        norbAutoencoder.buildModels()
        norbAutoencoder.summary()

        # Load network weights
        norbAutoencoder.loadWeights("./cacheWeights/" + config.stringDescriptor + ".weights.h5")

        # Obtain models
        encoder = norbAutoencoder.encoder()
        autoencoder = norbAutoencoder.autoencoder()
        decoder = norbAutoencoder.decoder()

        # Obtain datasets and carry out normalisation
        norbLoader = ScaleBetweenZeroAndOne(NorbLoader('./res/norb'), 0, 255)
        (xTrain, yTrain), _, (xTest, yTest) = norbLoader.loadData()

        # Random Sampling
        numSamplesSqrt = 10
        randomSamples = sp.randn(numSamplesSqrt * numSamplesSqrt, config.latentRepresentationDimension)
        randomDecoded = decoder.predict(randomSamples, batch_size=numSamplesSqrt*numSamplesSqrt)
        plt.figure(figsize=(numSamplesSqrt, numSamplesSqrt))
        for i in range(numSamplesSqrt * numSamplesSqrt):
            num = plt.subplot(numSamplesSqrt, numSamplesSqrt, i + 1)
            plt.imshow(randomDecoded[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
        plt.savefig("./out/" + config.stringDescriptor + "_random_sampling.png")

        # Display reconstructions
        numberExamplesReconstructions = 100
        xTestReconstructedExamples = autoencoder.predict(xTest[0:numberExamplesReconstructions], batch_size=numberExamplesReconstructions)
        plt.figure(figsize=(2, numberExamplesReconstructions))
        for i in range(numberExamplesReconstructions):
            num = plt.subplot(numberExamplesReconstructions, 2, 2 * i + 1)
            plt.imshow(xTest[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)

            num = plt.subplot(numberExamplesReconstructions, 2, 2 * i + 2)
            plt.imshow(xTestReconstructedExamples[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
        plt.savefig("./out/" + config.stringDescriptor + "_test_reconstructions.png")

        # Display reconstructions
        xTrainReconstructionExamples = autoencoder.predict(xTrain[0:numberExamplesReconstructions], batch_size=numberExamplesReconstructions)
        plt.figure(figsize=(2, numberExamplesReconstructions))
        for i in range(numberExamplesReconstructions):
            num = plt.subplot(numberExamplesReconstructions, 2, 2 * i + 1)
            plt.imshow(xTrain[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)

            num = plt.subplot(numberExamplesReconstructions, 2, 2 * i + 2)
            plt.imshow(xTrainReconstructionExamples[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
        plt.savefig("./out/" + config.stringDescriptor + "_train_reconstructions.png")
