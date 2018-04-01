import matplotlib
matplotlib.use('Agg')

from experiment.development.mnistTransformed.CreateMnistTransformedInterpolationDatasetExperiment import CreateMnistTransformedInterpolationDatasetExperiment

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
    experiment = CreateMnistTransformedInterpolationDatasetExperiment()

    experiment.run()
