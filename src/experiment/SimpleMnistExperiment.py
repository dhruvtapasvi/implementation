import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from dataset.basicLoader.MnistLoader import MnistLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from experiment.Experiment import Experiment
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig


class SimpleMnistExperiment(Experiment):
    def run(self):
        config = ConvolutionalAutoencoderConfig("./config/model/convolutional/mnist_conv_3_8_256_2_bce.json")

        mnistAutoencoder = config.fromConfig()
        mnistAutoencoder.buildModels()
        mnistAutoencoder.summary()

        batchSize = 100
        epochs = 50
        mnistLoader = ScaleBetweenZeroAndOne(MnistLoader(), 0, 255)
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = mnistLoader.loadData()
        print(x_train.shape, x_val.shape, x_test.shape)
        mnistAutoencoder.train(x_train, x_val, epochs, batchSize)

        # display a 2D plot of the digit classes in the latent space
        encoder = mnistAutoencoder.encoder()
        x_test_encoded = encoder.predict(x_test, batch_size=batchSize)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
        plt.colorbar()
        plt.savefig("out/" + config.stringDescriptor + '_fig1.png')

        # display a 2D manifold of the digits
        generator = mnistAutoencoder.decoder()
        generator.summary()
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
        # to produce values of the latent variables z, since the prior of the latent space is Gaussian
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                digit = (generator.predict(z_sample))[0]
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig("out/" + config.stringDescriptor + '_fig2.png')
