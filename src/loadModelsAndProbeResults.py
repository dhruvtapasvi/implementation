import dataset.loaderPackaged as loaders
import config.packagedConfigs as configs

from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.LoadModelExperiment import LoadModelExperiment
from experiment.ReconstructionsExperiment import ReconstructionsExperiment
from experiment.InterpolateExperiment import InterpolateExperiment

from interpolate.InterpolateLatentSpace import InterpolateLatentSpace


NUM_RECONSTRUCTIONS = 100
SQRT_NUM_SAMPLES = 10

configDatasetTuples = [
    (configs.conv_64x64_7_16_256_32_bce, loaders.shapesTransformedPackage)
]

for config, loaderPackage in configDatasetTuples:
    datasetModelPath = loaderPackage.name + "_" + config.stringDescriptor

    variationalAutoencoderBuilder = BuildModelExperiment(config)
    variationalAutoencoder = variationalAutoencoderBuilder.run()

    loadWeights = LoadModelExperiment(datasetModelPath, variationalAutoencoder)
    loadWeights.run()

    createReconstructions = ReconstructionsExperiment(loaderPackage.datasetLoader, config, variationalAutoencoder, NUM_RECONSTRUCTIONS, SQRT_NUM_SAMPLES, datasetModelPath)
    createReconstructions.run()

    interpolate = InterpolateLatentSpace(variationalAutoencoder)

    interpolateExperiment = InterpolateExperiment(loaderPackage.interpolateLoader, variationalAutoencoder, interpolate, datasetModelPath)
    interpolateExperiment.run()
