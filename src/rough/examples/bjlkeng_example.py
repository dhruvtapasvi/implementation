import matplotlib
matplotlib.use("Agg")

# Import libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import math
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

# Imports from project
from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne

# Set route to "implementation" folder
route = ".."
outRoute = route + "/outRough"
if not os.path.exists(outRoute):
    os.mkdir(outRoute)

# Load NORB
norbLoader = ScaleBetweenZeroAndOne(NorbLoader(route + "/res/norb"), 0, 255)
(xTrain, yTrain), (xVal, yVal), (xTest, yTest) = norbLoader.loadData()
norbShape = (96, 96)

# Load PCA and define PCA transform functions
pca = pickle.load(open(route + "/pca/norb_pca_500.p", "rb"))
n_components = pca.n_components_

def pcaForwardTransform(xData):
    return pca.transform(xData.reshape((xData.shape[0], -1)))

def pcaInverseTransform(xDataPca):
    xDataFlat = np.clip(pca.inverse_transform(xDataPca), 0, 0.999)
    xData = xDataFlat.reshape((xDataFlat.shape[0],) + norbShape)
    return xData

# Transform NORB dataset to PCA
xTrainPca = pcaForwardTransform(xTrain)
xValPca = pcaForwardTransform(xVal)
xTestPca = pcaForwardTransform(xTest)

# Create convenient data splits iterable list
dataSplits = [
    (xTrain, xTrainPca, yTrain, "train"),
    (xVal, xValPca, yVal, "val"),
    (xTest, xTestPca, yTest, "test")
]

# Display logic
def displayArraySamplesSideBySide(arrays, numSamplesToDisplay, fileName):
    numArrays = len(arrays)
    plt.figure(figsize=(numArrays, numSamplesToDisplay))
    for i in range(numSamplesToDisplay):
        for index, array in enumerate(arrays):
            num = plt.subplot(numSamplesToDisplay, numArrays, numArrays * i + index + 1)
            plt.imshow(array[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
    plt.savefig(fileName)

# Test PCA and inverse transform
numTestSamples = 10
for (xData, xDataPca, _, splitName) in dataSplits:
    displayArraySamplesSideBySide(
        [xData[0:numTestSamples], pcaInverseTransform(xDataPca[0:numTestSamples])],
        numTestSamples,
        outRoute + "/testPCAAndInverse_" + splitName + ".png")

# Define network and training parameters
batch_size = 1000
latent_dim = 512
intermediate_dim = 2048
epsilon_std = 1.0
epochs = 500
activation = 'relu'
dropout = 0.0
var_epsilon = 0.00001
learning_rate = 0.001

# Encoder network
x = Input(shape=(n_components,)) #Modified from original

batch_1 = BatchNormalization()(x)
hidden1_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")(batch_1)
hidden1_batch = BatchNormalization()(hidden1_dense)
hidden1 = Activation(activation)(hidden1_batch)
dropout_1 = Dropout(dropout)(hidden1)

hidden2_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")(dropout_1)
hidden2_batch = BatchNormalization()(hidden2_dense)
hidden2 = Activation(activation)(hidden2_batch)
dropout_2 = Dropout(dropout)(hidden2)

hidden3_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")(dropout_2)
hidden3_batch = BatchNormalization()(hidden3_dense)
hidden3 = Activation(activation)(hidden3_batch)
dropout_3 = Dropout(dropout)(hidden3)

z_mean_1_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")(dropout_3)
z_mean_1_batch = BatchNormalization()(z_mean_1_dense)
z_mean_1 = Activation(activation)(z_mean_1_batch)
z_mean_dropout_1 = Dropout(dropout)(z_mean_1)

z_mean_2_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")(z_mean_dropout_1)
z_mean_2_batch = BatchNormalization()(z_mean_2_dense)
z_mean_2 = Activation(activation)(z_mean_2_batch)
z_mean_dropout_2 = Dropout(dropout)(z_mean_2)
z_mean = Dense(latent_dim)(z_mean_dropout_2)

z_log_var_1_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")(dropout_3)
z_log_var_1_batch = BatchNormalization()(z_log_var_1_dense)
z_log_var_1 = Activation(activation)(z_log_var_1_batch)
z_log_var_dropout_1 = Dropout(dropout)(z_log_var_1)

z_log_var_2_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")(z_log_var_dropout_1)
z_log_var_2_batch = BatchNormalization()(z_log_var_2_dense)
z_log_var_2 = Activation(activation)(z_log_var_2_batch)
z_log_var_dropout_2 = Dropout(dropout)(z_log_var_2)
z_log_var = Dense(latent_dim)(z_log_var_dropout_2)

# Reparameterisation trick
def sampling(args, latent_dim=latent_dim, epsilon_std=epsilon_std):
    z_mean, z_log_var = args

    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), # modified from original
                              mean=0., stddev=epsilon_std)

    return z_mean + K.exp(z_log_var) * epsilon


z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# Decoder network
decoder_1_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")
decoder_1_batch = BatchNormalization()
decoder_1 = Activation(activation)
decoder_dropout_1 = Dropout(dropout)

decoder_2_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")
decoder_2_batch = BatchNormalization()
decoder_2 = Activation(activation)
decoder_dropout_2 = Dropout(dropout)

decoder_3_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")
decoder_3_batch = BatchNormalization()
decoder_3 = Activation(activation)
decoder_dropout_3 = Dropout(dropout)

decoder_4_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")
decoder_4_batch = BatchNormalization()
decoder_4 = Activation(activation)
decoder_dropout_4 = Dropout(dropout)

decoder_5_dense = Dense(intermediate_dim, kernel_initializer="he_normal", bias_initializer="uniform")
decoder_5_batch = BatchNormalization()
decoder_5 = Activation(activation)
decoder_dropout_5 = Dropout(dropout)

x_decoded_mean = Dense(n_components)

decoder_6_dense = Dense(n_components)
decoder_6_batch = BatchNormalization()
decoder_6 = Activation(activation)
decoder_dropout_6 = Dropout(dropout)
x_decoded_var = Dense(n_components, activation='relu', kernel_initializer="he_normal", bias_initializer="uniform")

_decoder_1 = decoder_dropout_1(decoder_1(decoder_1_batch(decoder_1_dense(z))))
_decoder_2 = decoder_dropout_2(decoder_2(decoder_2_batch(decoder_2_dense(_decoder_1))))
_decoder_3 = decoder_dropout_3(decoder_3(decoder_3_batch(decoder_3_dense(_decoder_2))))
_decoder_4 = decoder_dropout_4(decoder_4(decoder_4_batch(decoder_4_dense(_decoder_3))))
_decoder_5 = decoder_dropout_5(decoder_5(decoder_5_batch(decoder_5_dense(_decoder_4))))

_x_decoded_mean = x_decoded_mean(_decoder_5)

_decoder_6 = decoder_dropout_6(decoder_6(decoder_6_batch(decoder_6_dense(_decoder_5))))
_x_decoded_var = x_decoded_var(_decoder_6)
_output = _x_decoded_mean

# Loss Function
def kl_loss(x, x_decoded_mean):
    kl_loss = - 0.5 * K.sum(1. + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

    return K.mean(kl_loss)


def logx_loss(x, x_decoded_mean):
    loss = (0.5 * math.log(2 * math.pi)
            + 0.5 * K.log(_x_decoded_var + var_epsilon)
            + 0.5 * K.square(x - x_decoded_mean) / (_x_decoded_var + var_epsilon))
    loss = K.sum(loss, axis=-1)
    return K.mean(loss)


def vae_loss(x, x_decoded_mean):
    return logx_loss(x, x_decoded_mean) + kl_loss(x, x_decoded_mean)

# Compile Model
vae = Model(x, _output)
optimizer = Adam(lr=learning_rate)
vae.compile(optimizer=optimizer, loss=vae_loss,
            metrics=[logx_loss ,kl_loss])

vae.summary()

# Fit Model
start = time.time()

# early_stopping = EarlyStopping('loss', min_delta=0.1, patience=100)

# Modified from original:
history = vae.fit(
    x=xTrainPca,
    y=xTrainPca,
    batch_size=batch_size,
    epochs=epochs,
    # callbacks=[early_stopping],
    validation_data=(xValPca, xValPca)
)
# The author instead had:
# history = vae.fit_generator(
#     data_generator(X_train, pca, batch_size),
#     steps_per_epoch=len(X_train) // batch_size,
#     epochs=epochs,
#     validation_data=(xValPca, xValPca),
#     callbacks=[early_stopping],
#     verbose=0
# )

done = time.time()
elapsed = done - start
print("Elapsed: ", elapsed)

# Breakdown loss components:
vae_kl = Model(x, _output)
vae_kl.compile(optimizer='rmsprop', loss=kl_loss)
kl = vae_kl.evaluate(xTestPca, xTestPca, batch_size=batch_size)

vae_logx = Model(x, _output)
vae_logx.compile(optimizer='rmsprop', loss=logx_loss)
logx = vae_logx.evaluate(xTestPca, xTestPca, batch_size=batch_size)

print()
print("KL loss: {}".format(kl))
print("xent loss: {}".format(logx))

# Generator model
decoder_input = Input(shape=(latent_dim,))

_decoder_1 = decoder_dropout_1(decoder_1(decoder_1_batch(decoder_1_dense(decoder_input))))
_decoder_2 = decoder_dropout_2(decoder_2(decoder_2_batch(decoder_2_dense(_decoder_1))))
_decoder_3 = decoder_dropout_3(decoder_3(decoder_3_batch(decoder_3_dense(_decoder_2))))
_decoder_4 = decoder_dropout_4(decoder_4(decoder_4_batch(decoder_4_dense(_decoder_3))))
_decoder_5 = decoder_dropout_5(decoder_5(decoder_5_batch(decoder_5_dense(_decoder_4))))
_decoder_output = x_decoded_mean(_decoder_5)

generator = Model(decoder_input, _decoder_output)

# Save models and history
vae_path = outRoute + "/vae.hdf5"
vae.save(vae_path)

generator_path = outRoute + "/generator.hdf5"
generator.save(generator_path)

pickle.dump(history.history, open(outRoute + "/trainingHistory.p", "wb"))

# Test dataset reconstructions
numReconstructions = 20
for xData, xDataPca, _, splitName in dataSplits:
    xDataTruncated = xData[0:numReconstructions]
    xDataPcaTruncated = xDataPca[0:numReconstructions]
    xDataPcaInverted = pcaInverseTransform(xDataPcaTruncated)
    xDataPcaReconstructed = vae.predict(xDataPcaTruncated, numReconstructions)
    xDataPcaReconstructedInverted = pcaInverseTransform(xDataPcaReconstructed)
    displayArraySamplesSideBySide(
        [xDataTruncated, xDataPcaInverted, xDataPcaReconstructedInverted],
        numReconstructions,
        outRoute + "/vaeReconstructions_" + splitName + ".png")
