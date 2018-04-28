import numpy as np
from typing import List, Tuple
import pickle
import os


from config.routes import getUserStudyRoute
from display.printRows import printRows
from experiment.Experiment import Experiment
from model.VariationalAutoencoder import VariationalAutoencoder
from interpolate.Interpolate import Interpolate
from interpolate.InterpolateLatentSpace import InterpolateLatentSpace
from dataset.interpolate.process.CombineInterpolateLoaders import CombineInterpolateLoaders


RECONSTRUCTION_CATEGORIES_MULTIPLICITY = 8


INTERPOLATION_CATEGORIES_MULTIPLICITY = 8


class UserStudyGenerateImagesExperiment(Experiment):
    def __init__(
            self,
            numStudiesToMake,
            dataSplits,
            interpolationSubdatasets,
            datasetName: str,
            variationalAutoencoderConfigNamePairs: List[Tuple[VariationalAutoencoder, str]]
    ):
        self.__numStudiesToMake = numStudiesToMake
        _, _, (self.__xTest, _) = dataSplits
        combineInterpolateLoaders = CombineInterpolateLoaders()
        self.__interpolationSubdataset = combineInterpolateLoaders.combine(interpolationSubdatasets, "Combined")
        self.__folderRoute = getUserStudyRoute(datasetName + "/")
        self.__architectures = variationalAutoencoderConfigNamePairs
        self.__interpolate = Interpolate()
        if not os.path.isdir(self.__folderRoute):
            os.mkdir(self.__folderRoute)

    def run(self):
        self.__saveReconstructions()
        self.__saveInterpolations()

    def __saveReconstructions(self):
        numRandomTestSelections = self.__numStudiesToMake * RECONSTRUCTION_CATEGORIES_MULTIPLICITY
        randomTestSelectionsIndices = np.random.choice(len(self.__xTest), numRandomTestSelections, replace=False)
        randomTestSelections = self.__xTest[randomTestSelectionsIndices]

        reconstructions = {
            "positiveControl": randomTestSelections,
            "randomImage": np.random.random_sample(randomTestSelections.shape)
        }

        for variationalAutoencoder, configName in self.__architectures:
            reconstructions[configName] = variationalAutoencoder.autoencoder().predict(randomTestSelections, batch_size=100)

        combinedTestSelections = np.concatenate([randomTestSelections for _ in range(len(reconstructions))])
        combinedNames = []
        combinedTestSelectionsReconstructionsList = []
        for key, value in reconstructions.items():
            combinedNames += [key for _ in range(numRandomTestSelections)]
            combinedTestSelectionsReconstructionsList.append(value)
        combinedTestSelectionsReconstructions = np.concatenate(combinedTestSelectionsReconstructionsList)

        self.__saveArrayPictures([combinedTestSelections, combinedTestSelectionsReconstructions], combinedNames, "reconstruction")

    def __saveInterpolations(self):
        numRandomInterpolationSelections = self.__numStudiesToMake * INTERPOLATION_CATEGORIES_MULTIPLICITY
        randomInterpolationSelectionsIndices = np.random.choice(len(self.__interpolationSubdataset.xLeft), numRandomInterpolationSelections, replace=False)

        randomInterpolationXLeft = self.__interpolationSubdataset.xLeft[randomInterpolationSelectionsIndices]
        randomInterpolationXRight = self.__interpolationSubdataset.xRight[randomInterpolationSelectionsIndices]
        randomInterpolationXCentre = self.__interpolationSubdataset.xCentre[randomInterpolationSelectionsIndices]

        interpolations = {
            "positiveControl": randomInterpolationXCentre,
            "randomImage": np.random.random_sample(randomInterpolationXLeft.shape),
            "left": randomInterpolationXLeft,
            "interpolateImageSpace": self.__interpolate.interpolateAll(randomInterpolationXLeft, randomInterpolationXRight, 2)[:, 1]
        }

        for variationalAutoencoder, configName in self.__architectures:
            interpolateLatentSpace = InterpolateLatentSpace(variationalAutoencoder)
            _, interpolatedReconstructed = interpolateLatentSpace.interpolateAll(randomInterpolationXLeft, randomInterpolationXRight, 2)
            interpolations["interpolateLatentSpace_" + configName] = interpolatedReconstructed[:, 1]

        combinedLeftSelections = np.concatenate([randomInterpolationXLeft for _ in range(len(interpolations))])
        combinedRightSelections = np.concatenate([randomInterpolationXRight for _ in range(len(interpolations))])
        combinedNames = []
        combinedTestSelectionsReconstructionsList = []
        for key, value in interpolations.items():
            combinedNames += [key for _ in range(numRandomInterpolationSelections)]
            combinedTestSelectionsReconstructionsList.append(value)
        combinedTestSelectionsReconstructions = np.concatenate(combinedTestSelectionsReconstructionsList)

        self.__saveArrayPictures([combinedLeftSelections, combinedTestSelectionsReconstructions, combinedRightSelections], combinedNames, "interpolation")

    def __saveArrayPictures(self, arrays, names, dirName):
        directory = self.__folderRoute + dirName + "/"
        if not os.path.isdir(directory):
            os.mkdir(directory)
        fileNameStem = directory + "image_"

        arraysLength = len(names)
        randomShuffle = np.random.permutation(np.arange(arraysLength))

        arraysRandomlyShuffled = [array[randomShuffle] for array in arrays]
        namesRandomlyShuffled = [names[index]for index in randomShuffle]

        printRows(arraysRandomlyShuffled, arraysLength, fileNameStem)
        with open(directory + "labels.p", "wb") as file:
            pickle.dump(namesRandomlyShuffled, file)
