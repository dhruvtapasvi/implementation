from config.ConfigParser import ConfigParser
from abc import ABCMeta

from model.loss.binaryCrossEntropyLoss import binaryCrossEntropyLossConstructor
from model.loss.meanSquaredErrorLoss import meanSquaredErrorLossConstructor


class VaeConfig(ConfigParser, metaclass=ABCMeta):
    def __init__(self, parameters):
        self.__setParameters(parameters)

    def __setParameters(self, parameters):
        self.__reconstructionLossConstructor = binaryCrossEntropyLossConstructor \
            if parameters["reconstructionLoss"] == "binaryCrossEntropy" else meanSquaredErrorLossConstructor
        self.__klLossWeight = parameters["klLossWeight"]
        self.__inputRepresentationDimensions = tuple(parameters["inputRepresentationDimensions"])
        self.__latentRepresentationDimension = parameters["latentRepresentationDimension"]
        self.__descriptor = parameters["descriptor"]

    @property
    def reconstructionLossConstructor(self):
        return self.__reconstructionLossConstructor

    @property
    def klLossWeight(self):
        return self.__klLossWeight

    @property
    def inputRepresentationDimensions(self):
        return self.__inputRepresentationDimensions

    @property
    def latentRepresentationDimension(self):
        return self.__latentRepresentationDimension

    @property
    def stringDescriptor(self):
        return self.__descriptor
