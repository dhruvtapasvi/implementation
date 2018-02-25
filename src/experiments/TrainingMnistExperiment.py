from datasets.basicLoaders.MnistLoader import MnistLoader
from experiments.Experiment import Experiment
from model.ConvolutionalAutoencoder import ConvolutionalAutoencoder


class TrainingMnistExperiment(Experiment):
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

        # Obtain datasets and carry out normalisation
        mnistLoader = MnistLoader()
        (xTrain, yTrain), (xTest, yTrain) = mnistLoader.loadData()
        print(xTrain.shape, xTest.shape)
        xTrain = xTrain.astype('float32') / 255.
        xTest = xTest.astype('float32') / 255.

        # Train model
        batchSize = 100
        epochs = 50
        mnistConvolutionalAutoencoder.train(xTrain, xTest, epochs, batchSize)

        # Save network weights:
        mnistConvolutionalAutoencoder.saveWeights("convolutionalTrainingWeights.h5")
