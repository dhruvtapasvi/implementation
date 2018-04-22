from evaluation.results.ResultsStore import ResultsStore


class DictionaryResultsStore(ResultsStore):
    def __init__(self):
        self.__dict = {}

    def storeValue(self, keysList, value):
        dict = self.__dict
        for key in keysList[:-1]:
            dict = dict.setdefault(key, {})
        dict[keysList[-1]] = value

    def getValue(self, keysList):
        value = self.__dict
        for key in keysList:
            value = value.get(key)
        return value

    def getDictionary(self):
        return self.__dict
