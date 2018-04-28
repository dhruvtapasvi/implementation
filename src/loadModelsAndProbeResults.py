from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.LoadModelExperiment import LoadModelExperiment
from experiment.ReconstructionsExperiment import ReconstructionsExperiment
from experiment.InterpolateExperiment import InterpolateExperiment

from interpolate.InterpolateLatentSpace import InterpolateLatentSpace

from experiment.experimentalConfigTuples import experimentalConfigTuples as prepackagedExperimentalTuples


experimentalTuples = prepackagedExperimentalTuples

NUM_RECONSTRUCTIONS = 100
SQRT_NUM_SAMPLES = 10


for experimentalTuple in experimentalTuples:
    variationalAutoencoderBuilder = BuildModelExperiment(experimentalTuple.config)
    variationalAutoencoder = variationalAutoencoderBuilder.run()

    loadWeights = LoadModelExperiment(experimentalTuple.stringDescriptor, variationalAutoencoder)
    loadWeights.run()

    createReconstructions = ReconstructionsExperiment(experimentalTuple.datasetPackage.datasetLoader, experimentalTuple.config, variationalAutoencoder, NUM_RECONSTRUCTIONS, SQRT_NUM_SAMPLES, experimentalTuple.stringDescriptor)
    createReconstructions.run()

    interpolate = InterpolateLatentSpace(variationalAutoencoder)

    interpolateExperiment = InterpolateExperiment(experimentalTuple.datasetPackage.interpolateLoader, variationalAutoencoder, interpolate, experimentalTuple.stringDescriptor)
    interpolateExperiment.run()
