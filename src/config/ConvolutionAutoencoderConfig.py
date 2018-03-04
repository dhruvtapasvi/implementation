import json
from config.ConfigParser import ConfigParser

from model.loss.binaryCrossEntropyLoss import binaryCrossEntropyLossConstructor
from model.loss.meanSquaredErrorLoss import meanSquaredErrorLossConstructor
from model.architecture.ConvolutionalAutoencoder import ConvolutionalAutoencoder

class ConvolutionalAutoencoderConfig(ConfigParser):
    def __init__(self, file):
        with open(file) as file:
            parameters = json.load(file)
            self.__setParameters(parameters)

    def __setParameters(self, parameters):
        self.__reconstructionLossConstructor = binaryCrossEntropyLossConstructor \
            if parameters["reconstructionLoss"] == "binaryCrossEntropy" else meanSquaredErrorLossConstructor
        self.__klLossWeight = parameters["klLossWeight"]
        self.__inputRepresentationDimensions = tuple(parameters["inputRepresentationDimensions"])
        self.__numberConvolutions = parameters["numberConvolutions"]
        self.__baseConvolutionalDepth = parameters["baseConvolutionalDepth"]
        self.__intermediateRepresentationDimension = parameters["intermediateRepresentationDimension"]
        self.__latentRepresentationDimension = parameters["latentRepresentationDimension"]
        self.__descriptor = parameters["descriptor"]

    def stringDescriptor(self):
        return self.__descriptor

    def fromConfig(self):
        return ConvolutionalAutoencoder(
            self.__reconstructionLossConstructor,
            self.__klLossWeight,
            self.__inputRepresentationDimensions,
            self.__numberConvolutions,
            self.__baseConvolutionalDepth,
            self.__intermediateRepresentationDimension,
            self.__latentRepresentationDimension
        )
