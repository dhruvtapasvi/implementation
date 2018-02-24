import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from experiments.Experiment import Experiment
from model.ConvolutionalAutoencoder import ConvolutionalAutoencoder
from datasets.MnistLoader import MnistLoader


class UseLoadedWeightsExperiment(Experiment):
    def run(self):
        # Hyperparameters
        originalImageDimensions = (28, 28)
        intermediateDimension = 256
        latentDimension = 2
        numConvolutions = 3
        baseConvolutionalDepth = 8

        # Build model and exhibit summary
        mnistConvolutionalAutoencoder = ConvolutionalAutoencoder(
            originalImageDimensions,
            numConvolutions,
            baseConvolutionalDepth,
            intermediateDimension,
            latentDimension
        )
        mnistConvolutionalAutoencoder.buildModels()
        mnistConvolutionalAutoencoder.summary()

        # Load network weights
        mnistConvolutionalAutoencoder.loadWeights("convolutionalTrainingWeights.h5")

        # Obtain models
        encoder = mnistConvolutionalAutoencoder.encoder()
        autoencoder = mnistConvolutionalAutoencoder.autoencoder()
        decoder = mnistConvolutionalAutoencoder.decoder()

        # Obtain datasets and carry out normalisation
        mnistLoader = MnistLoader()
        (xTrain, yTrain), (xTest, yTest) = mnistLoader.loadData()
        xTest = xTest.astype('float32') / 255.

        # Display the latent space:
        batchSizeLatentSpace = 100
        xTestEncodedExamples = encoder.predict(xTest, batch_size=batchSizeLatentSpace)
        plt.figure(figsize=(6,6))
        plt.scatter(xTestEncodedExamples[:, 0], xTestEncodedExamples[:, 1], c=yTest)
        plt.colorbar()
        plt.savefig('vae_fig1.png')

        # Display reconstructions
        numberExamplesReconstructions = 100
        autoencoder = mnistConvolutionalAutoencoder.autoencoder()
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
        plt.savefig('vae_fig2.png')

        # Random Sampling
        numSamplesSqrt = 10
        randomSamples = sp.randn(numSamplesSqrt * numSamplesSqrt, latentDimension)
        randomDecoded = decoder.predict(randomSamples, batch_size=numSamplesSqrt*numSamplesSqrt)
        plt.figure(figsize=(numSamplesSqrt, numSamplesSqrt))
        for i in range(numSamplesSqrt * numSamplesSqrt):
            num = plt.subplot(numSamplesSqrt, numSamplesSqrt, i + 1)
            plt.imshow(randomDecoded[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
        plt.savefig('vae_fig3.png')

        # Interpolation
        selectedPairs = [
            (5, 191),  # Right slanted vs left slanted 1
            (0, 243),  # 7 without horizontal bar vs with horiztonal bar
            (148, 311),  # Thin vs fat 0
            (59, 207),  # Uncurved vs very curved belly of 5
            (140, 259),  # Differing 6 styles
            (99, 17),  # 9 to 7
            (270, 52)  # 3 to 5
        ]
        numIntermediate = 8
        plt.figure(figsize=(numIntermediate + 2, len(selectedPairs)))
        for i, pair in enumerate(selectedPairs):
            firstIndex, secondIndex = pair
            firstRepresentation = xTest[firstIndex]
            secondRepresentation = xTest[secondIndex]
            latents = encoder.predict(np.array([firstRepresentation, secondRepresentation]), batch_size=2)
            firstLatent = latents[0]
            secondLatent = latents[1]
            print(firstLatent)
            print(secondLatent)

            divisor = numIntermediate - 1
            interpolatedLatents = np.array([
                firstLatent * (divisor - x) / divisor + secondLatent * x / divisor for x in range(numIntermediate)
            ])
            interpolatedRepresentations = decoder.predict(interpolatedLatents, batch_size=numIntermediate)
            num = plt.subplot(len(selectedPairs), numIntermediate + 2, i * (numIntermediate + 2) + 1)
            plt.imshow(firstRepresentation)
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)

            for j in range(numIntermediate):
                num = plt.subplot(len(selectedPairs), numIntermediate + 2, i * (numIntermediate + 2) + j + 2)
                plt.imshow(interpolatedRepresentations[j])
                plt.gray()
                num.get_xaxis().set_visible(False)
                num.get_yaxis().set_visible(False)

            num = plt.subplot(len(selectedPairs), numIntermediate + 2, i * (numIntermediate + 2) + numIntermediate + 2)
            plt.imshow(secondRepresentation)
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
        plt.savefig('vae_fig4.png')
