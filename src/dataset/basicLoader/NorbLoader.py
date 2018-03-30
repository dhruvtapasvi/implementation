import numpy as np

from dataset.DatasetLoader import DatasetLoader
from dataset.assemble.NorbAssembler import NorbAssembler
from dataset.parse.NorbParser import NorbParser
from dataset.process.FilterDatasetLabelPredicate import FilterDatasetLabelPredicate
import dataset.info.NorbInfo as norbInfo


class NorbLoader(DatasetLoader):
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
        print("Loading Norb...")
        XTrain, YTrain = self.__loadData(False)
        XTest, YTest = self.__loadData(True)
        datasetFilter = FilterDatasetLabelPredicate()

        test, (XRemaining, YRemaining) =\
            datasetFilter.split(XTrain, YTrain, lambda row: row[norbInfo.NorbLabelIndex.INSTANCE.value] >= norbInfo.NORB_TEST_INSTANCES)
        validation, (XExtraTrain, YExtraTrain) =\
            datasetFilter.split(XRemaining, YRemaining, lambda row: row[norbInfo.NorbLabelIndex.INSTANCE.value] >= norbInfo.NORB_VALIDATION_INSTANCES)
        train = np.concatenate((XTest, XExtraTrain)), np.concatenate((YTest, YExtraTrain))
        print("Norb Loaded!")
        return train, validation, test

    def dataPointShape(self):
        return norbInfo.NORB_IMAGE_DIMENSIONS, norbInfo.NORB_LABEL_DIMENSIONS
