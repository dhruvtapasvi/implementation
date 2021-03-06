import matplotlib
matplotlib.use('Agg')


import evaluation.results.packageResults as resultsStores
from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.LoadModelExperiment import LoadModelExperiment
from experiment.ReconstructionsExperiment import ReconstructionsExperiment
from experiment.InterpolateExperiment import InterpolateExperiment
from experiment.RecordLossesExperiment import RecordLossesExperiment
from experiment.SamplingExperiment import SamplingExperiment

from experiment.experimentalConfigTuples import experimentalConfigTuples as prepackagedExperimentalTuples


experimentalTuples = prepackagedExperimentalTuples
NUM_SAMPLE_RECONSTRUCTIONS = 10
NUM_SAMPLES = 10
NUM_SAMPLE_INTERPOLATIONS = 10
NUM_INTERPOLATION_INTERVALS = 6


for experimentalTuple in experimentalTuples:
    variationalAutoencoderBuilder = BuildModelExperiment(experimentalTuple.config)
    variationalAutoencoder = variationalAutoencoderBuilder.run()

    loadWeights = LoadModelExperiment(experimentalTuple.stringDescriptor, variationalAutoencoder)
    loadWeights.run()

    dataSplits = experimentalTuple.datasetPackage.datasetLoader.loadData()
    interpolationSplits = experimentalTuple.datasetPackage.interpolateLoader.loadInterpolationData()
    dataSplitsName = experimentalTuple.datasetPackage.name
    configName = experimentalTuple.config.stringDescriptor

    recordLossesExperiment = RecordLossesExperiment(dataSplits, dataSplitsName, variationalAutoencoder, configName, resultsStores.modelLossResults)
    recordLossesExperiment.run()

    reconstructionsExperiment = ReconstructionsExperiment(dataSplits, variationalAutoencoder, NUM_SAMPLE_RECONSTRUCTIONS, experimentalTuple.stringDescriptor)
    reconstructionsExperiment.run()

    samplingExperiment = SamplingExperiment(dataSplits, experimentalTuple.config, variationalAutoencoder, NUM_SAMPLES, experimentalTuple.stringDescriptor)
    samplingExperiment.run()

    interpolateExperiment = InterpolateExperiment(
        interpolationSplits,
        dataSplitsName,
        variationalAutoencoder,
        configName,
        experimentalTuple.metricLatentSpace,
        experimentalTuple.metricImageSpace,
        experimentalTuple.stringDescriptor,
        resultsStores.interpolationResults,
        NUM_SAMPLE_INTERPOLATIONS,
        NUM_INTERPOLATION_INTERVALS
    )
    interpolateExperiment.run()
