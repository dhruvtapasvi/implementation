import matplotlib
matplotlib.use('Agg')


from experiment.TrainingMnistExperiment import TrainingMnistExperiment
from experiment.UseLoadedWeightsExperiment import UseLoadedWeightsExperiment
from experiment.SimpleMnistExperiment import SimpleMnistExperiment


if __name__ == '__main__':
    experiment = SimpleMnistExperiment()
    # experiment = TrainingMnistExperiment()
    # experiment = UseLoadedWeightsExperiment()
    experiment.run()
