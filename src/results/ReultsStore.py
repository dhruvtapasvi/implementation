from abc import ABCMeta, abstractclassmethod


class ResultsStore(metaclass=ABCMeta):
    @abstractclassmethod
    def storeValue(self, keysList, value):
        raise NotImplementedError

    @abstractclassmethod
    def getValue(self, keysList):
        raise NotImplementedError


    @abstractclassmethod
    def getDictionary(self):
        raise NotImplementedError
