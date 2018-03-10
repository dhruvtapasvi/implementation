import matplotlib
matplotlib.use('Agg')

from experiment.mnist.SimpleMnistExperiment import SimpleMnistExperiment
from experiment.mnist.TrainingMnistExperiment import TrainingMnistExperiment
from experiment.mnist.UseLoadedWeightsMnistExperiment import UseLoadedWeightsExperiment

from experiment.norb.TrainingNorbExperiment import TrainingNorbExperiment

if __name__ == '__main__':
    # experiment = SimpleMnistExperiment()
    # experiment = TrainingMnistExperiment()
    # experiment = UseLoadedWeightsExperiment()
    experiment = TrainingNorbExperiment()
    experiment.run()
