import json
from config.VaeConfig import VaeConfig

from model.architecture.ConvolutionalDeepIntermediateAutoencoder import ConvolutionalDeepIntermediateAutoencoder


class ConvolutionalDeepIntermediateAutoencoderConfig(VaeConfig):
    def __init__(self, file):
        with open(file) as file:
            parameters = json.load(file)
            super().__init__(parameters)
            self.__setParameters(parameters)

    def __setParameters(self, parameters):
        self.__numberConvolutions = parameters["numberConvolutions"]
        self.__baseConvolutionalDepth = parameters["baseConvolutionalDepth"]
        self.__downsampleLast = parameters["downSampleLast"]
        self.__encoderIntermediateDimensions = parameters["encoderIntermediateDimensions"]
        self.__decoderIntermediateDimensions = parameters["decoderIntermediateDimensions"]

    @property
    def numberConvolutions(self):
        return self.__numberConvolutions

    @property
    def baseConvolutionalDepth(self):
        return self.__baseConvolutionalDepth

    @property
    def downsampleLast(self):
        return self.__downsampleLast

    @property
    def encoderIntermediateDimensions(self):
        return self.__encoderIntermediateDimensions

    @property
    def decoderIntermediateDimensions(self):
        return self.__decoderIntermediateDimensions

    def fromConfig(self):
        return ConvolutionalDeepIntermediateAutoencoder(
            self.reconstructionLossConstructor,
            self.klLossWeight,
            self.inputRepresentationDimensions,
            self.numberConvolutions,
            self.downsampleLast,
            self.baseConvolutionalDepth,
            self.encoderIntermediateDimensions,
            self.decoderIntermediateDimensions,
            self.latentRepresentationDimension
        )
