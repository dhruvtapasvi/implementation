from dataset.interpolate.basic.MnistTransformedInterpolateLoader import MnistTransformedInterpolateLoader
from experiment.Experiment import Experiment


class InterpolateMnistTransformedExperiment(Experiment):
    def run(self):
        mnistTransformedInterpolateLoader = MnistTransformedInterpolateLoader("./res/mnistTransformedInterpolate")
        (xInterpolate, yInterpolate) = mnistTransformedInterpolateLoader.loadInterpolationData()
        print(xInterpolate.shape)
        print(yInterpolate.shape)
