import matplotlib
matplotlib.use('Agg')

from experiment.development.mnist.UseLoadedWeightsMnistExperiment import UseLoadedWeightsExperiment
from experiment.development.mnist.TransformMnistExperiment import TransformMnistExperiment

from experiment.development.norb.NorbInterpolationExperiment import NorbInterpolationExperiment

from experiment.development.mnistTransformed.TrainingMnistTransformedExperiment import TrainingMnistTransformedExperiment
from experiment.development.mnistTransformed.UseLoadedWeightsMnistTransformedExperiment import UseLoadedWeightsMnistTransformedExperiment

if __name__ == '__main__':
    # experiment = SimpleMnistExperiment()
    # experiment = TrainingMnistExperiment()
    # experiment = UseLoadedWeightsExperiment()
    # experiment = TransformMnistExperiment()

    # experiment = TrainingNorbExperiment()
    # experiment = UseLoadedWeightsNorbExperiment()
    # experiment = NorbInterpolationExperiment()

    experiment = TrainingMnistTransformedExperiment()
    experiment = UseLoadedWeightsMnistTransformedExperiment()

    experiment.run()
