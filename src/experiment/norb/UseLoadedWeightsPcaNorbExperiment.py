import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from experiment.Experiment import Experiment
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne

from model.loss.meanSquaredErrorLoss import meanSquaredErrorLossConstructor
from model.architecture.PcaAutoencoder import PcaAutoencoder
import pickle


class UseLoadedWeightsPcaNorbExperiment(Experiment):
    def __predictConstructor(self, decoder, pca):
        def predict(examples, batch_size):
            decoded = decoder.predict(examples, batch_size)
            recon = np.clip(pca.inverse_transform(decoded), 0.0, 0.999)
            reconOrigShape = recon.reshape((recon.shape[0], 96, 96))
            return reconOrigShape

        return predict

    def run(self):
        config = {  }
        config["stringDescriptor"] = "norb_pca_500_512_4_128_0"

        # Build model and exhibit summary
        reconstructionLossConstructor = meanSquaredErrorLossConstructor
        klLossWeight = 1.0
        inputRepresentationDimensions = (500,)
        intermediateRepresentationDimension = 512
        numIntermediateDimensions = 4
        latentRepresentationDimension = 128
        dropout = 0.0
        norbAutoencoder = PcaAutoencoder(reconstructionLossConstructor, klLossWeight, inputRepresentationDimensions, intermediateRepresentationDimension, numIntermediateDimensions,latentRepresentationDimension, dropout)
        norbAutoencoder.buildModels()
        norbAutoencoder.summary()

        # Load network weights
        norbAutoencoder.loadWeights("./cacheWeights/" + config["stringDescriptor"] + ".weights.h5")

        # Obtain models
        encoder = norbAutoencoder.encoder()
        autoencoder = norbAutoencoder.autoencoder()
        decoder = norbAutoencoder.decoder()

        # Obtain datasets and carry out normalisation and PCA
        norbLoader = ScaleBetweenZeroAndOne(NorbLoader('./res/norb'), 0, 255)
        (xTrain, yTrain), _, (xTest, yTest) = norbLoader.loadData()
        pca = pickle.load(open("./pca/norb_pca_500.p", "rb"))
        xTrainPca = pca.transform(xTrain.reshape((xTrain.shape[0],-1)))
        xTestPca = pca.transform(xTest.reshape((xTest.shape[0],-1)))

        predictDecoder = self.__predictConstructor(decoder, pca)
        predictAutoencoder = self.__predictConstructor(autoencoder, pca)


        # Random Sampling
        numSamplesSqrt = 10
        randomSamples = sp.randn(numSamplesSqrt * numSamplesSqrt, latentRepresentationDimension)
        randomDecoded = predictDecoder(randomSamples, batch_size=numSamplesSqrt*numSamplesSqrt)
        plt.figure(figsize=(numSamplesSqrt, numSamplesSqrt))
        for i in range(numSamplesSqrt * numSamplesSqrt):
            num = plt.subplot(numSamplesSqrt, numSamplesSqrt, i + 1)
            plt.imshow(randomDecoded[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
        plt.savefig("./out/" + config["stringDescriptor"] + "_random_sampling.png")

        # Display reconstructions
        numberExamplesReconstructions = 100
        xTestReconstructedExamples = predictAutoencoder(xTestPca[0:numberExamplesReconstructions], batch_size=numberExamplesReconstructions)
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
        plt.savefig("./out/" + config["stringDescriptor"] + "_test_reconstructions.png")

        # Display reconstructions
        xTrainReconstructionExamples = predictAutoencoder(xTrainPca[0:numberExamplesReconstructions], batch_size=numberExamplesReconstructions)
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
        plt.savefig("./out/" + config["stringDescriptor"] + "_train_reconstructions.png")
