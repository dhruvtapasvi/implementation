import json
from config.VaeConfig import VaeConfig

from model.architecture.DenseAutoencoder import DenseAutoencoder


class DenseAutoencoderConfig(VaeConfig):
    def __init__(self, file):
        with open(file) as file:
            parameters = json.load(file)
            super().__init__(parameters)
            self.__setParameters(parameters)

    def __setParameters(self, parameters):
        self.__intermediateRepresentationDimension = parameters["intermediateRepresentationDimension"]

    @property
    def intermediateRepresentationDimension(self):
        return self.__intermediateRepresentationDimension

    def fromConfig(self):
        return DenseAutoencoder(
            self.reconstructionLossConstructor,
            self.klLossWeight,
            self.inputRepresentationDimensions,
            self.intermediateRepresentationDimension,
            self.latentRepresentationDimension
        )
