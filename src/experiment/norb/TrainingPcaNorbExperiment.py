import pickle

from experiment.Experiment import Experiment
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne

from model.loss.meanSquaredErrorLoss import meanSquaredErrorLossConstructor
from model.architecture.PcaAutoencoderFittedVariance import PcaAutoencoderFittedVariance
from model.architecture.PcaAutoencoder import PcaAutoencoder


class TrainingPcaNorbExperiment(Experiment):
    def run(self):
        config = {  }
        config["stringDescriptor"] = "norb_pca_500_2048_1_128_0_fitted_variance"

        # Build model and exhibit summary
        reconstructionLossConstructor = meanSquaredErrorLossConstructor
        klLossWeight = 1.0
        inputRepresentationDimensions = (500,)
        intermediateRepresentationDimension = 2048
        numIntermediateDimensions = 1
        latentRepresentationDimension = 128
        dropout = 0.0
        norbAutoencoder = PcaAutoencoderFittedVariance(reconstructionLossConstructor, klLossWeight, inputRepresentationDimensions, intermediateRepresentationDimension, numIntermediateDimensions,latentRepresentationDimension, dropout)
        norbAutoencoder.buildModels()
        norbAutoencoder.summary()

        # Obtain datasets and carry out normalisation and pca
        norbLoader = ScaleBetweenZeroAndOne(NorbLoader("../res/norb"), 0, 255)
        (xTrain, _), (xVal, _), _ = norbLoader.loadData()
        pca500 = pickle.load(open("../pca/norb_pca_500.p", "rb"))
        xTrain = pca500.transform(xTrain.reshape((xTrain.shape[0], -1)))
        xVal = pca500.transform(xVal.reshape((xVal.shape[0], -1)))

        print(xTrain.shape)


        # Train model
        batchSize = 1000
        epochs = 300
        trainingHistory = norbAutoencoder.train(xTrain, xVal, epochs, batchSize)

        # Save training history and network weights:
        pickle.dump(trainingHistory.history, open(config["stringDescriptor"] + ".history.p", "wb"))
        norbAutoencoder.saveWeights(config["stringDescriptor"] + ".weights.h5")
