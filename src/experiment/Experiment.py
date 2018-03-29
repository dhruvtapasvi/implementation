from abc import ABCMeta, abstractclassmethod


class Experiment(metaclass=ABCMeta):
    @abstractclassmethod
    def run(self):
        """
        The experiment should take as constructor arguments all the relevant objects needed to carry out the experiment
        :return: The results of running the experiment, for example a trained model. This can then be passed on to the
        next experiment that needs it in the constructor argument.
        """
        raise NotImplementedError
