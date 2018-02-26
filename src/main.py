import matplotlib
matplotlib.use('Agg')


from experiments.TrainingMnistExperiment import TrainingMnistExperiment
from experiments.UseLoadedWeightsExperiment import UseLoadedWeightsExperiment
from experiments.SimpleMnistExperiment import SimpleMnistExperiment


if __name__ == '__main__':
    # experiment = SimpleMnistExperiment()
    # experiment = TrainingMnistExperiment()
    experiment = UseLoadedWeightsExperiment()
    experiment.run()
