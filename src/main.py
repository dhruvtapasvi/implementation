import matplotlib
matplotlib.use('Agg')

from experiment.development.norb.NorbInterpolationExperiment import NorbInterpolationExperiment

if __name__ == '__main__':
    # experiment = SimpleMnistExperiment()
    # experiment = TrainingMnistExperiment()
    # experiment = UseLoadedWeightsExperiment()
    # experiment = TrainingNorbExperiment()
    # experiment = UseLoadedWeightsNorbExperiment()
    experiment = NorbInterpolationExperiment()
    experiment.run()
