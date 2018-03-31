from dataset.basicLoader.MnistTransformedLoader import MnistTransformedLoader
from experiment.Experiment import Experiment


class TrainingMnistTransformedExperiment(Experiment):
    def run(self):
        mnistLoader = MnistTransformedLoader("./res/mnistTransformed_2")
        (_, yTrain), _, _ = mnistLoader.loadData()
        print(yTrain[0:10])
