from dataset.basicLoader.MnistLoader import MnistLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from experiment.Experiment import Experiment
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig


class TrainingMnistExperiment(Experiment):
    def run(self):
        config = ConvolutionalAutoencoderConfig("./config/model/convolutional/mnist_conv_bce_2.json")

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
        mnistAutoencoder.train(xTrain, xValidation, epochs, batchSize)

        # Save network weights:
        mnistAutoencoder.saveWeights(config.stringDescriptor() + "_trainingWeights.h5")
