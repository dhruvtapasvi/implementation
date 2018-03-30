from abc import ABCMeta, abstractclassmethod


class Experiment(metaclass=ABCMeta):
    @abstractclassmethod
    def run(self):
        """
        The experiment should take as constructor arguments all the relevant objects needed to carry out the experiment.
        Experiment instances should put together all the required results for the dissertation, so that they combine all
        the disparate sections of the code by stitching them together in a cohesive flow.
        Experiments can call other sub-experiments, to make things more convenient
        main.py should stitch together the experiments, and the final main.py file should contain all the experiments
        required to produce all the final results of the dissertation.
        The experiments with other aspects e.g. hyperparameter search, which don't produce results included in the
        dissertation can go inside uncalled experiment classes.
        :return: The results of running the experiment, for example a trained model. They can then be passed on to the
        next experiment that needs them in its constructor argument.
        """
        raise NotImplementedError
