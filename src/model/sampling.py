from keras.backend import random_normal, exp, shape


unitStdDev = 1.0


def sampleConstructor(dimension):
    def sample(args):
        latentMean, latentVariance = args
        epsilon = random_normal(shape=(shape(latentMean)[0], dimension), mean=0., stddev=unitStdDev)
        return latentMean + exp(latentVariance / 2) * epsilon
    return sample