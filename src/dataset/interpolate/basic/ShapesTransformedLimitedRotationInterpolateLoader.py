import numpy as np

from dataset.loader.DatasetLoader import DatasetLoader
from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.interpolate.process.CreateTransformedInterpolateData import CreateTransformedInterpolateData
import dataset.info.ShapesInfoLimitedRotation as shapesInfoLimitedRotation
from dataset.interpolate.process.CombineInterpolateLoaders import CombineInterpolateLoaders


class ShapesTransformedLimitedRotationInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, baseShapesLoader: DatasetLoader):
        self.__combineInterpolateLoaders = CombineInterpolateLoaders()
        self.__loaders = []

        self.__rotationsLoaders = []
        for shearFactor in np.linspace(-shapesInfoLimitedRotation.TRANSFORM_SHEAR_FACTOR, shapesInfoLimitedRotation.TRANSFORM_SHEAR_FACTOR, shapesInfoLimitedRotation.DEFAULT_SHEAR_NUM_DELTAS + 1):
            for enlargementFactor in np.linspace(-shapesInfoLimitedRotation.TRANSFORM_LOG2_STRETCH_FACTOR, shapesInfoLimitedRotation.TRANSFORM_LOG2_STRETCH_FACTOR, shapesInfoLimitedRotation.DEFAULT_ENLARGEMENT_NUM_DELTAS + 1):
                defaultInterpolationFactors = (shapesInfoLimitedRotation.DEFAULT_ROTATION_FACTOR, shearFactor, enlargementFactor)
                self.__rotationsLoaders.append(CreateTransformedInterpolateData(
                    baseShapesLoader,
                    shapesInfoLimitedRotation.PADDING,
                    *defaultInterpolationFactors,
                    *shapesInfoLimitedRotation.INTERPOLATE_JOINT_FACTORS
                ))

        self.__shearsLoaders = []
        for rotAngle in np.linspace(*shapesInfoLimitedRotation.TRANSFORM_MIN_MAX_ROTATIONS,shapesInfoLimitedRotation.DEFAULT_ROTATION_NUM_DELTAS + 1):
            for enlargementFactor in np.linspace(-shapesInfoLimitedRotation.TRANSFORM_LOG2_STRETCH_FACTOR,shapesInfoLimitedRotation.TRANSFORM_LOG2_STRETCH_FACTOR,shapesInfoLimitedRotation.DEFAULT_ENLARGEMENT_NUM_DELTAS + 1):
                defaultInterpolationFactors = (rotAngle, shapesInfoLimitedRotation.DEFAULT_SHEAR_FACTOR, enlargementFactor)
                self.__shearsLoaders.append(CreateTransformedInterpolateData(
                    baseShapesLoader,
                    shapesInfoLimitedRotation.PADDING,
                    *defaultInterpolationFactors,
                    *shapesInfoLimitedRotation.INTERPOLATE_JOINT_FACTORS
                ))

        self.__enlargementLoaders = []
        for rotAngle in np.linspace(*shapesInfoLimitedRotation.TRANSFORM_MIN_MAX_ROTATIONS, shapesInfoLimitedRotation.DEFAULT_ROTATION_NUM_DELTAS + 1):
            for shearFactor in np.linspace(-shapesInfoLimitedRotation.TRANSFORM_SHEAR_FACTOR, shapesInfoLimitedRotation.TRANSFORM_SHEAR_FACTOR, shapesInfoLimitedRotation.DEFAULT_SHEAR_NUM_DELTAS + 1):
                defaultInterpolationFactors = (rotAngle, shearFactor, shapesInfoLimitedRotation.DEFAULT_LOG2_STRETCH_FACTOR)
                self.__enlargementLoaders.append(CreateTransformedInterpolateData(
                    baseShapesLoader,
                    shapesInfoLimitedRotation.PADDING,
                    *defaultInterpolationFactors,
                    *shapesInfoLimitedRotation.INTERPOLATE_JOINT_FACTORS
                ))

    def loadInterpolationData(self):
        finalSubdatasets = []
        for index, loaders in enumerate([self.__rotationsLoaders, self.__shearsLoaders, self.__enlargementLoaders]):
            loadersData = [loader.loadInterpolationData() for loader in loaders]
            relevantLoadersData = list(zip(*loadersData))[index]
            combinedRelevantLoadersData = self.__combineInterpolateLoaders.combine(relevantLoadersData)
            finalSubdatasets.append(combinedRelevantLoadersData)
        return finalSubdatasets

    def dataPointShape(self):
        return shapesInfoLimitedRotation.IMAGE_DIMENSIONS, shapesInfoLimitedRotation.LABEL_DIMENSIONS
