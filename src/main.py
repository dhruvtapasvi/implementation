import matplotlib
matplotlib.use('Agg')


from experiments.TrainingMnistExperiment import TrainingMnistExperiment
from experiments.UseLoadedWeightsMnistExperiment import UseLoadedWeightsExperiment
from experiments.TrainingNorbExperiment import TrainingNorbExperiment
from experiments.UseLoadedWeightsNorbExperiment import UseLoadedWeightsNorbExperiment


if __name__ == '__main__':
    # experiment = TrainingMnistExperiment()
    # experiment = UseLoadedWeightsExperiment()
    # experiment = TrainingNorbExperiment()
    experiment = UseLoadedWeightsNorbExperiment()
    experiment.run()
