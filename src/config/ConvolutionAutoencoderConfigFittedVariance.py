import json
from config.VaeConfig import VaeConfig

from model.architecture.ConvolutionalAutoencoderFittedVariance import ConvolutionalAutoencoderFittedVariance


class ConvolutionalAutoencoderConfigFittedVariance(VaeConfig):
    def __init__(self, file):
        with open(file) as file:
            parameters = json.load(file)
            super().__init__(parameters)
            self.__setParameters(parameters)

    def __setParameters(self, parameters):
        self.__numberConvolutions = parameters["numberConvolutions"]
        self.__baseConvolutionalDepth = parameters["baseConvolutionalDepth"]
        self.__intermediateRepresentationDimension = parameters["intermediateRepresentationDimension"]

    @property
    def numberConvolutions(self):
        return self.__numberConvolutions

    @property
    def baseConvolutionalDepth(self):
        return self.__baseConvolutionalDepth

    @property
    def intermediateRepresentationDimension(self):
        return self.__intermediateRepresentationDimension

    def fromConfig(self):
        return ConvolutionalAutoencoderFittedVariance(
            self.reconstructionLossConstructor,
            self.klLossWeight,
            self.inputRepresentationDimensions,
            self.numberConvolutions,
            self.baseConvolutionalDepth,
            self.intermediateRepresentationDimension,
            self.latentRepresentationDimension
        )
