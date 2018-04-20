import json
from config.VaeConfig import VaeConfig

from model.architecture.DeepDenseAutoencoder import DeepDenseAutoencoder


class DeepDenseAutoencoderConfig(VaeConfig):
    def __init__(self, file):
        with open(file) as file:
            parameters = json.load(file)
            super().__init__(parameters)
            self.__setParameters(parameters)

    def __setParameters(self, parameters):
        self.__encoderDimensions = parameters["encoderDimensions"]
        self.__decoderDimensions = parameters["decoderDimensions"]

    @property
    def encoderDimensions(self):
        return self.__encoderDimensions

    @property
    def decoderDimensions(self):
        return self.__decoderDimensions

    def fromConfig(self):
        return DeepDenseAutoencoder(
            self.reconstructionLossConstructor,
            self.klLossWeight,
            self.inputRepresentationDimensions,
            self.encoderDimensions,
            self.__decoderDimensions,
            self.latentRepresentationDimension
        )
