import matplotlib
matplotlib.use('Agg')

from experiment.development.norb.NorbInterpolationExperiment import NorbInterpolationExperiment

from experiment.InterpolateExperiment import InterpolateExperiment
from dataset.loader.basic.MnistLoader import MnistLoader
from dataset.interpolate.process.CreateTransformedInterpolateData import CreateTransformedInterpolateData
import dataset.info.MnistTransformedInfo as mnistTransformedInfo

if __name__ == '__main__':
    # experiment = SimpleMnistExperiment()
    # experiment = TrainingMnistExperiment()
    # experiment = UseLoadedWeightsExperiment()
    # experiment = TransformMnistExperiment()

    # experiment = TrainingNorbExperiment()
    # experiment = UseLoadedWeightsNorbExperiment()
    experiment = NorbInterpolationExperiment()

    # experiment = TrainingMnistTransformedExperiment()
    # experiment = UseLoadedWeightsMnistTransformedExperiment()
    # experiment = CreateMnistTransformedInterpolationDatasetExperiment()
    # experiment = InterpolateMnistTransformedExperiment()

    experiment = InterpolateExperiment(CreateTransformedInterpolateData(MnistLoader(), mnistTransformedInfo.PADDING, *mnistTransformedInfo.DEFAULT_JOINT_FACTORS, *mnistTransformedInfo.INTERPOLATE_JOINT_FACTORS))

    experiment.run()
