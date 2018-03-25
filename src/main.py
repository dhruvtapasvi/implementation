import matplotlib
matplotlib.use('Agg')

from experiment.mnist.SimpleMnistExperiment import SimpleMnistExperiment
from experiment.mnist.TrainingMnistExperiment import TrainingMnistExperiment
from experiment.mnist.UseLoadedWeightsMnistExperiment import UseLoadedWeightsExperiment

from experiment.norb.TrainingNorbExperiment import TrainingNorbExperiment
from experiment.norb.UseLoadedWeightsNorbExperiment import UseLoadedWeightsNorbExperiment
from experiment.norb.TrainingPcaNorbExperiment import TrainingPcaNorbExperiment
from experiment.norb.UseLoadedWeightsPcaNorbExperiment import UseLoadedWeightsPcaNorbExperiment
from experiment.norb.TrainingNorbExperimentFittedVariance import TrainingNorbExperimentFittedVariance
from experiment.norb.UseLoadedWeightsNorbExperimentFittedVariance import UseLoadedWeightsNorbExperimentFittedVariance


if __name__ == '__main__':
    # experiment = SimpleMnistExperiment()
    # experiment = TrainingMnistExperiment()
    # experiment = UseLoadedWeightsExperiment()
    # experiment = TrainingNorbExperiment()
    # experiment = UseLoadedWeightsNorbExperiment()
    experiment = TrainingPcaNorbExperiment()
    # experiment = UseLoadedWeightsPcaNorbExperiment()
    # experiment = TrainingNorbExperimentFittedVariance()
    # experiment = UseLoadedWeightsNorbExperimentFittedVariance()
    experiment.run()
