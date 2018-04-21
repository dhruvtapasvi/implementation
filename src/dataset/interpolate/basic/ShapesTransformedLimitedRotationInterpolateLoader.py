import numpy as np

from dataset.loader.DatasetLoader import DatasetLoader
from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.interpolate.process.CreateTransformedInterpolateData import CreateTransformedInterpolateData
import dataset.info.ShapesInfoLimitedRotation as shapesInfoLimitedRotation


class ShapesTransformedLimitedRotationInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, baseShapesLoader: DatasetLoader):
        self.__mnistLoader = CreateTransformedInterpolateData(
            baseShapesLoader,
            shapesInfoLimitedRotation.PADDING,
            *shapesInfoLimitedRotation.DEFAULT_JOINT_FACTORS,
            *shapesInfoLimitedRotation.INTERPOLATE_JOINT_FACTORS
        )

    def loadInterpolationData(self):
        return self.__mnistLoader.loadInterpolationData()

    def dataPointShape(self):
        return shapesInfoLimitedRotation.IMAGE_DIMENSIONS, shapesInfoLimitedRotation.LABEL_DIMENSIONS
