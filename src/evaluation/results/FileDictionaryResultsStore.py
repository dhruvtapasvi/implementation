import pickle
import os

from evaluation.results.ResultsStore import ResultsStore


class FileDictionaryResultsStore(ResultsStore):
    def __init__(self, dictionaryLocation: str):
        self.__dictionaryLocation = dictionaryLocation
        if not os.path.isfile(self.__dictionaryLocation):
            self.__storeDict({})

    def getValue(self, keysList):
        value = self.getDictionary()
        for key in keysList:
            value = value[key]
        return value

    def getDictionary(self):
        with open(self.__dictionaryLocation, "rb") as file:
            return pickle.load(file)

    def storeValue(self, keysList, value):
        dict = self.getDictionary()
        nestedDict = dict
        for key in keysList[:-1]:
            nestedDict = nestedDict.setdefault(key, {})
        nestedDict[keysList[-1]] = value
        self.__storeDict(dict)

    def __storeDict(self, dict):
        with open(self.__dictionaryLocation, "wb") as file:
            return pickle.dump(dict, file)
