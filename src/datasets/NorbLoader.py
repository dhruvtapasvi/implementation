from datasets.DatasetLoader import DatasetLoader
from parse.NorbParser import NorbParser
import numpy as np


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

    def _loadData(self, isTestData: bool = False):
        norbRoot = self._norbPath + '_' + (self._norbTestLabel if isTestData else self._norbTrainLabel) + '_'
        readFlag = 'r'
        with open(norbRoot + self._imagePostfix, readFlag) as imageFile:
            with open(norbRoot + self._categoryPostfix, readFlag) as categoryFile:
                with open(norbRoot + self._infoPostfix, readFlag) as infoFile:
                    images = self._norbParser.parse(imageFile).data
                    categories = np.array([self._norbParser.parse(categoryFile).data])
                    details = self._norbParser.parse(infoFile).data
                    return images, np.concatenate((categories.T, details), 1)

    def loadData(self):
        return self._loadData(), self._loadData(True)
