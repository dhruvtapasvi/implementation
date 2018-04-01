import matplotlib
matplotlib.use('Agg')

from experiment.development.mnistTransformed.CreateMnistTransformedInterpolationDatasetExperiment import CreateMnistTransformedInterpolationDatasetExperiment
from experiment.development.mnistTransformed.InterpolateMnistTransformedExperiment import InterpolateMnistTransformedExperiment

if __name__ == '__main__':
    # experiment = SimpleMnistExperiment()
    # experiment = TrainingMnistExperiment()
    # experiment = UseLoadedWeightsExperiment()
    # experiment = TransformMnistExperiment()

    # experiment = TrainingNorbExperiment()
    # experiment = UseLoadedWeightsNorbExperiment()
    # experiment = NorbInterpolationExperiment()

    # experiment = TrainingMnistTransformedExperiment()
    # experiment = UseLoadedWeightsMnistTransformedExperiment()
    # experiment = CreateMnistTransformedInterpolationDatasetExperiment()
    experiment = InterpolateMnistTransformedExperiment()

    experiment.run()
