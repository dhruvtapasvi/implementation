import matplotlib
matplotlib.use('Agg')


from experiments.TrainingMnistExperiment import TrainingMnistExperiment
from experiments.UseLoadedWeightsExperiment import UseLoadedWeightsExperiment


if __name__ == '__main__':
    # simpleMnistExperiment = TrainingMnistExperiment()
    useLoadedWeightsExperiment = UseLoadedWeightsExperiment()
    useLoadedWeightsExperiment.run()
