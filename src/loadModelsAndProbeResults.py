import dataset.loaderPackaged as loaders
import config.packagedConfigs as configs
import evaluation.results.packageResults as resultsStores

from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.LoadModelExperiment import LoadModelExperiment
from experiment.ReconstructionsExperiment import ReconstructionsExperiment
from experiment.InterpolateExperiment import InterpolateExperiment
from experiment.RecordLossesExperiment import RecordLossesExperiment
from experiment.SamplingExperiment import SamplingExperiment

from evaluation.metric.SquaredError import SquaredError
from evaluation.metric.BinaryCrossEntropy import BinaryCrossEntropy

from interpolate.InterpolateLatentSpace import InterpolateLatentSpace

from experiment.ExperimentalConfigTuple import ExperimentalConfigTuple
from experiment.experimentalConfigTuples import experimentalConfigTuples as prepackagedExperimentalTuples


experimentalTuples = [
    ExperimentalConfigTuple(loaders.shapesTransformedPackage, configs.conv_64x64_7_16_256_32_bce, 1000, 0, SquaredError(1), BinaryCrossEntropy())
]


NUM_RECONSTRUCTIONS = 100
SQRT_NUM_SAMPLES = 10


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

    reconstructionsExperiment = ReconstructionsExperiment(dataSplits, variationalAutoencoder, 10, experimentalTuple.stringDescriptor)
    reconstructionsExperiment.run()

    samplingExperiment = SamplingExperiment(dataSplits, experimentalTuple.config, variationalAutoencoder, 10, experimentalTuple.stringDescriptor)
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
        10,
        6
    )
    interpolateExperiment.run()
