from abc import ABCMeta, abstractclassmethod


class Experiment(metaclass=ABCMeta):
    @abstractclassmethod
    def run(self):
        """
        The experiment should take as constructor arguments all the relevant objects needed to carry out the experiment.
        Experiment instances should combine all the disparate sections of the code by stitching them together in a cohesive flow.
        main.py or other top level scripts should stitch together the experiments, so that running the scripts produces the required results for this dissertation
        The experiments with other aspects e.g. hyperparameter search, which don't produce results directly included in the dissertation can go inside uncalled experiment classes.
        :return: The results of running the experiment, for example a trained model. They can then be passed on to the next experiment that needs them in its constructor argument.
        """
        raise NotImplementedError
