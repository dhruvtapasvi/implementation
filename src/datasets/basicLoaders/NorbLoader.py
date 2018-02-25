import numpy as np

from datasets.DatasetLoader import DatasetLoader
from datasets.assemble.NorbAssembler import NorbAssembler
from datasets.parse.NorbParser import NorbParser
from datasets.process.FilterDatasetLabelPredicate import FilterDatasetLabelPredicate


class NorbLoader(DatasetLoader):
    __NORB_VALIDATION_INSTANCES = 7
    __NORB_TEST_INSTANCES = 9
    __NORB_INSTANCE_ATTRIBUTE = 2

    def __init__(
            self,
            norbHome,
            norbPrefix='norb',
            norbTrainLabel='train',
            norbTestLabel='test',
            imagePostfix='image',
            categoryPostfix='category',
            infoPostfix='info'):
        self._norbPath = norbHome + ('' if norbHome[-1] == '/' else '/') + norbPrefix
        self._norbTrainLabel = norbTrainLabel
        self._norbTestLabel = norbTestLabel
        self._imagePostfix = imagePostfix
        self._categoryPostfix = categoryPostfix
        self._infoPostfix = infoPostfix
        self._norbParser = NorbParser()
        self._norbAssembler = NorbAssembler()

    def __loadData(self, isTestData: bool = False):
        norbRoot = self._norbPath + '_' + (self._norbTestLabel if isTestData else self._norbTrainLabel) + '_'
        readFlag = 'r'
        with open(norbRoot + self._imagePostfix, readFlag) as imageFile:
            with open(norbRoot + self._categoryPostfix, readFlag) as categoryFile:
                with open(norbRoot + self._infoPostfix, readFlag) as infoFile:
                    images = self._norbParser.parse(imageFile).data
                    categories = np.array([self._norbParser.parse(categoryFile).data])
                    details = self._norbParser.parse(infoFile).data
                    return self._norbAssembler.assemble(images, np.concatenate((categories.T, details), 1))

    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        XTrain, YTrain = self.__loadData(False)
        XTest, YTest = self.__loadData(True)
        datasetFilter = FilterDatasetLabelPredicate()

        test, (XRemaining, YRemaining) =\
            datasetFilter.split(XTrain, YTrain, lambda row: row[NorbLoader.__NORB_INSTANCE_ATTRIBUTE] >= NorbLoader.__NORB_TEST_INSTANCES)
        validation, (XExtraTrain, YExtraTrain) =\
            datasetFilter.split(XRemaining, YRemaining, lambda row: row[NorbLoader.__NORB_INSTANCE_ATTRIBUTE] >= NorbLoader.__NORB_VALIDATION_INSTANCES)
        train = np.concatenate((XTest, XExtraTrain)), np.concatenate((YTest, YExtraTrain))
        return train, validation, test
