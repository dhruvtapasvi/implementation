import matplotlib
matplotlib.use('Agg')

from keras.layers import Input, Conv2D, Deconv2D, Lambda
from keras.models import Model
from model.sampling import samplingConstructor
from model.VariationalAutoencoderLossOld import VariationalAutoencoderLossOld
from datasets.MnistLoader import MnistLoader
import matplotlib.pyplot as plt

originalImageDimensions = (28, 28, 1)
baseConvolutionalDepth = 8
intermediateDimension = 100
latentDimension = 10

inputLayer = Input(shape=originalImageDimensions)

# 28 * 28 * 1

convolutionalDepth = baseConvolutionalDepth
intermediateLayer = Conv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(inputLayer)
intermediateLayer = Conv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)
intermediateLayer = Conv2D(convolutionalDepth, (2, 2), strides=(2, 2), activation='relu', padding='valid')(intermediateLayer)

# 14 * 14 * baseConvolutionalDepth

convolutionalDepth *= 2
intermediateLayer = Conv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)
intermediateLayer = Conv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)
intermediateLayer = Conv2D(convolutionalDepth, (2, 2), strides=(2, 2), activation='relu', padding='valid')(intermediateLayer)

# 7 * 7 * 2 baseConvolutionalDepth
intermediateLayer = Conv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)
intermediateLayer = Conv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)

intermediateLayer = Conv2D(intermediateDimension, (7, 7), activation='relu', padding='valid')(intermediateLayer)

# 1 * 1 * intermediateDimension

latentMean = Conv2D(latentDimension, (1, 1), activation='relu', padding='valid')(intermediateLayer)
latentVariance = Conv2D(latentDimension, (1, 1), activation='relu', padding='valid')(intermediateLayer)
latentLayer = Lambda(samplingConstructor(latentDimension), output_shape=(1, 1, latentDimension))([latentMean, latentVariance])

# 1 * 1 * latentDimension

intermediateLayer = Deconv2D(intermediateDimension, (1, 1), activation='relu', padding='valid')(latentLayer)

# 1 * 1 * intermediateDimension

intermediateLayer = Deconv2D(convolutionalDepth, (7, 7), activation='relu', padding='valid')(intermediateLayer)

# 7 * 7 * 2 baseConvolutionalDepth

intermediateLayer = Deconv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)
intermediateLayer = Deconv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)
intermediateLayer = Deconv2D(convolutionalDepth, (2, 2), strides=(2, 2), activation='relu', padding='valid')(intermediateLayer)

# 14 * 14 * 2 baseConvolutionalDepth

intermediateLayer = Deconv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)
convolutionalDepth //= 2
intermediateLayer = Deconv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)

# 14 * 14 * baseConvolutionalDepth

intermediateLayer = Deconv2D(convolutionalDepth, (2, 2), strides=(2, 2), activation='relu', padding='valid')(intermediateLayer)

# 28 * 28 * baseConvolutionalDepth

intermediateLayer = Deconv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)
convolutionalDepth = 1
outputLayer = Deconv2D(convolutionalDepth, (3, 3), activation='relu', padding='same')(intermediateLayer)
lossLayer = VariationalAutoencoderLossOld(28 * 28, latentMean, latentVariance)([inputLayer, outputLayer])


variationalAutoencoder = Model(inputLayer, lossLayer)
variationalAutoencoder.summary()
variationalAutoencoder.compile(optimizer='rmsprop', loss=None)


mnistLoader = MnistLoader()
(x_train, y_train), (x_test, y_test) = mnistLoader.loadData()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)

variationalAutoencoder.fit(x_train, shuffle=True, epochs=60, batch_size=100, validation_data=(x_test, None))

# build a model to project inputs on the latent space
encoder = Model(inputLayer, latentMean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=100)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0, 0, 0], x_test_encoded[:, 0, 0, 1], c=y_test)
plt.colorbar()
plt.savefig('vae_fig1.png')
