from datasets.basicLoaders.MnistLoader import MnistLoader
from datasets.preprocessLoaders.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
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
        mnistLoader = ScaleBetweenZeroAndOne(MnistLoader(), 0, 255)
        (xTrain, yTrain), (xValidation, yValidation), _ = mnistLoader.loadData()
        print(xTrain.shape, xValidation.shape)

        # Train model
        batchSize = 100
        epochs = 50
        mnistConvolutionalAutoencoder.train(xTrain, xValidation, epochs, batchSize)

        # Save network weights:
        mnistConvolutionalAutoencoder.saveWeights("convolutionalTrainingWeights.h5")
