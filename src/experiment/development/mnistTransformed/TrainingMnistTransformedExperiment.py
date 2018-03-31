from dataset.basicLoader.MnistTransformedLoader import MnistTransformedLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from experiment.Experiment import Experiment
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
import pickle


class TrainingMnistTransformedExperiment(Experiment):
    def run(self):
        rootPath = ".."

        config = ConvolutionalAutoencoderConfig(rootPath + "/config/model/convolutional/mnist_transformed_conv_7_8_256_10_bce.json")
        vae = config.fromConfig()
        vae.buildModels()
        vae.summary()

        mnistLoader = ScaleBetweenZeroAndOne(MnistTransformedLoader(rootPath + "/res/mnistTransformed_1"), 0, 255)
        (xTrain, yTrain), (xVal, yVal), _ = mnistLoader.loadData()

        # Train model
        batchSize = 1000
        epochs = 100
        trainingHistory = vae.train(xTrain, xVal, epochs, batchSize)

        # Save training history and network weights:
        pickle.dump(trainingHistory.history, open(rootPath + "/modelTrainingHistory/" + config.stringDescriptor + ".history.p", "wb"))
        vae.saveWeights(rootPath + "/cacheWeights/" + config.stringDescriptor + ".weights.h5")
