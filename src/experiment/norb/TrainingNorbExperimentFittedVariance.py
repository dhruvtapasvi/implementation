import pickle

from experiment.Experiment import Experiment
from config.ConvolutionAutoencoderConfigFittedVariance import ConvolutionalAutoencoderConfigFittedVariance
from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne


class TrainingNorbExperimentFittedVariance(Experiment):
    def run(self):
        config = ConvolutionalAutoencoderConfigFittedVariance("../config/model/convolutional/norb_conv_6_16_256_100_mse.json")

        # Build model and exhibit summary
        norbAutoencoder = config.fromConfig()
        norbAutoencoder.buildModels()
        norbAutoencoder.summary()

        # Obtain datasets and carry out normalisation
        norbLoader = ScaleBetweenZeroAndOne(NorbLoader("../res/norb"), 0, 255)
        (xTrain, _), (xVal, _), _ = norbLoader.loadData()

        # Train model
        batchSize = 100
        epochs = 50
        trainingHistory = norbAutoencoder.train(xTrain, xVal, epochs, batchSize)

        # Save training history and network weights:
        pickle.dump(trainingHistory.history, open(config.stringDescriptor + ".history.p", "wb"))
        norbAutoencoder.saveWeights(config.stringDescriptor + ".weights.h5")