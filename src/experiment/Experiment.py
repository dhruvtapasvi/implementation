from abc import ABCMeta, abstractclassmethod


class Experiment(metaclass=ABCMeta):
    @abstractclassmethod
    def run(self):
        raise NotImplementedError
