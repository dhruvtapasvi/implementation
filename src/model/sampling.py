from keras.backend import random_normal, exp, shape


unitStdDev = 1.0


def samplingConstructor(latentRepresentationDimension):
    def sampling(args):
        latentRepresentationMean, latentRepresentationLogVariance = args
        epsilon = random_normal(shape=(shape(latentRepresentationMean)[0], latentRepresentationDimension), mean=0., stddev=unitStdDev)
        return latentRepresentationMean + exp(latentRepresentationLogVariance / 2) * epsilon

    return sampling
