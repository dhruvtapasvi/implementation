import pickle

from results.ResultsStore import ResultsStore


class FileDictionaryResultsStore(ResultsStore):
    def __init__(self, dictionaryLocation: str):
        self.__dictionaryLocation = dictionaryLocation
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
        for key in keysList[:-1]:
            dict = dict.setdefault(key, {})
        dict[keysList[-1]] = value
        self.__storeDict(dict)

    def __storeDict(self, dict):
        with open(self.__dictionaryLocation, "wb") as file:
            return pickle.dump(dict, file)
