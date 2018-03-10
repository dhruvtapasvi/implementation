import matplotlib
matplotlib.use('Agg')

from experiment.mnist.UseLoadedWeightsMnistExperiment import UseLoadedWeightsExperiment

if __name__ == '__main__':
    # experiment = SimpleMnistExperiment()
    # experiment = TrainingMnistExperiment()
    experiment = UseLoadedWeightsExperiment()
    experiment.run()
