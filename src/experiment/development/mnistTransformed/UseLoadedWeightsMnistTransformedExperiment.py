import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from dataset.basicLoader.MnistTransformedLoader import MnistTransformedLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from experiment.Experiment import Experiment
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig


class UseLoadedWeightsMnistTransformedExperiment(Experiment):
    def run(self):
        config = ConvolutionalAutoencoderConfig("config/model/convolutional/mnist_transformed_conv_7_8_256_10_bce.json")

        mnistAutoencoder = config.fromConfig()
        mnistAutoencoder.buildModels()
        mnistAutoencoder.summary()

        # Load network weights
        mnistAutoencoder.loadWeights("cacheWeights/" + config.stringDescriptor + ".weights.h5")

        # Obtain models
        encoder = mnistAutoencoder.encoder()
        autoencoder = mnistAutoencoder.autoencoder()
        decoder = mnistAutoencoder.decoder()

        # Obtain datasets and carry out normalisation
        mnistLoader = ScaleBetweenZeroAndOne(MnistTransformedLoader("./res/mnistTransformed_10"), 0, 255)
        (xTrain, _), _, (xTest, _) = mnistLoader.loadData()

        # # Display the latent space:
        # batchSizeLatentSpace = 100
        # xTestEncodedExamples = encoder.predict(xTest, batch_size=batchSizeLatentSpace)
        # plt.figure(figsize=(6,6))
        # plt.scatter(xTestEncodedExamples[:, 0], xTestEncodedExamples[:, 1], c=yTest[:, 0])
        # plt.colorbar()
        # plt.savefig("out/" + config.stringDescriptor + '_fig1.png')

        # Display reconstructions
        numberExamplesReconstructions = 100
        autoencoder = mnistAutoencoder.autoencoder()
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
        plt.savefig("out2/" + config.stringDescriptor + '_testReconstructions.png')

        # Display reconstructions
        numberExamplesReconstructions = 100
        xTrainReconstructedExamples = autoencoder.predict(xTrain[0:numberExamplesReconstructions], batch_size=numberExamplesReconstructions)
        plt.figure(figsize=(2, numberExamplesReconstructions))
        for i in range(numberExamplesReconstructions):
            num = plt.subplot(numberExamplesReconstructions, 2, 2 * i + 1)
            plt.imshow(xTrain[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)

            num = plt.subplot(numberExamplesReconstructions, 2, 2 * i + 2)
            plt.imshow(xTrainReconstructedExamples[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
        plt.savefig("out2/" + config.stringDescriptor + '_trainReconstructions.png')

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
        plt.savefig("out2/" + config.stringDescriptor + '_randomSampling.png')

        # # Interpolation
        # selectedPairs = [
        #     (5, 191),  # Right slanted vs left slanted 1
        #     (0, 243),  # 7 without horizontal bar vs with horiztonal bar
        #     (148, 311),  # Thin vs fat 0
        #     (59, 207),  # Uncurved vs very curved belly of 5
        #     (140, 259),  # Differing 6 styles
        #     (99, 17),  # 9 to 7
        #     (270, 52)  # 3 to 5
        # ]
        # numIntermediate = 8
        # plt.figure(figsize=(numIntermediate + 2, len(selectedPairs)))
        # for i, pair in enumerate(selectedPairs):
        #     firstIndex, secondIndex = pair
        #     firstRepresentation = xTest[firstIndex]
        #     secondRepresentation = xTest[secondIndex]
        #     latents = encoder.predict(np.array([firstRepresentation, secondRepresentation]), batch_size=2)
        #     firstLatent = latents[0]
        #     secondLatent = latents[1]
        #     print(firstLatent)
        #     print(secondLatent)
        #
        #     divisor = numIntermediate - 1
        #     interpolatedLatents = np.array([
        #         firstLatent * (divisor - x) / divisor + secondLatent * x / divisor for x in range(numIntermediate)
        #     ])
        #     interpolatedRepresentations = decoder.predict(interpolatedLatents, batch_size=numIntermediate)
        #     num = plt.subplot(len(selectedPairs), numIntermediate + 2, i * (numIntermediate + 2) + 1)
        #     plt.imshow(firstRepresentation)
        #     plt.gray()
        #     num.get_xaxis().set_visible(False)
        #     num.get_yaxis().set_visible(False)
        #
        #     for j in range(numIntermediate):
        #         num = plt.subplot(len(selectedPairs), numIntermediate + 2, i * (numIntermediate + 2) + j + 2)
        #         plt.imshow(interpolatedRepresentations[j])
        #         plt.gray()
        #         num.get_xaxis().set_visible(False)
        #         num.get_yaxis().set_visible(False)
        #
        #     num = plt.subplot(len(selectedPairs), numIntermediate + 2, i * (numIntermediate + 2) + numIntermediate + 2)
        #     plt.imshow(secondRepresentation)
        #     plt.gray()
        #     num.get_xaxis().set_visible(False)
        #     num.get_yaxis().set_visible(False)
        # plt.savefig("out/" + config.stringDescriptor + '_fig4.png')
