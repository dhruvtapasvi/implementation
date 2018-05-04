from experiment.BuildModelExperiment import BuildModelExperiment

from experiment.experimentalConfigTuples import experimentalConfigTuples as prepackagedExperimentalTuples


experimentalTuples = prepackagedExperimentalTuples


for experimentalTuple in experimentalTuples:
    print(experimentalTuple.stringDescriptor)
    variationalAutoencoderBuilder = BuildModelExperiment(experimentalTuple.config)
    variationalAutoencoder = variationalAutoencoderBuilder.run()
