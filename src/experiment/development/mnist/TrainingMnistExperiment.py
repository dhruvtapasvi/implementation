import pickle

from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
from dataset.loader.basic.MnistLoader import MnistLoader
from dataset.loader.preprocess import ScaleBetweenZeroAndOne
from experiment.Experiment import Experiment


class TrainingMnistExperiment(Experiment):
    def run(self):
        config = ConvolutionalAutoencoderConfig("config/model/convolutional/mnist_conv_3_8_256_2_bce.json")

        mnistAutoencoder = config.fromConfig()
        mnistAutoencoder.buildModels()
        mnistAutoencoder.summary()

        # Obtain datasets and carry out normalisation
        mnistLoader = ScaleBetweenZeroAndOne(MnistLoader(), 0, 255)
        (xTrain, yTrain), (xValidation, yValidation), _ = mnistLoader.loadData()
        print(xTrain.shape, xValidation.shape)

        # Train model
        batchSize = 100
        epochs = 50
        trainingHistory = mnistAutoencoder.train(xTrain, xValidation, epochs, batchSize)

        # Save training history and network weights:
        pickle.dump(trainingHistory.history, open("./modelTrainingHistory/" + config.stringDescriptor + ".history.p", "wb"))
        mnistAutoencoder.saveWeights("cacheWeights/" + config.stringDescriptor + ".weights.h5")
