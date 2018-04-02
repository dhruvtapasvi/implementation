import matplotlib
matplotlib.use('Agg')

from experiment.development.norb.NorbInterpolationExperiment import NorbInterpolationExperiment

from experiment.final.InterpolateExperiment import InterpolateExperiment
from dataset.interpolate.basic.NorbInterpolateLoader import NorbInterpolateLoader
from dataset.interpolate.process.InstancesInterpolateLoader import InstancesInterpolateLoader
from dataset.loader.basic.NorbLoader import NorbLoader
from dataset.loader.basic.MnistLoader import MnistLoader
import dataset.info.MnistInfo as mnistInfo
from experiment.development.mnistTransformed.CreateMnistTransformedInterpolationDatasetExperiment import CreateMnistTransformedInterpolationDatasetExperiment
from experiment.development.mnistTransformed.InterpolateMnistTransformedExperiment import InterpolateMnistTransformedExperiment
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
