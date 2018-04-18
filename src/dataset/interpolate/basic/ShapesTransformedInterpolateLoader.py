import numpy as np

from dataset.loader.DatasetLoader import DatasetLoader
from dataset.interpolate.InterpolateDatasetLoader import InterpolateDatasetLoader
from dataset.interpolate.process.CreateTransformedInterpolateData import CreateTransformedInterpolateData
import dataset.info.ShapesInfo as shapesInfo


class ShapesTransformedInterpolateLoader(InterpolateDatasetLoader):
    def __init__(self, baseShapesLoader: DatasetLoader):
        self.__mnistLoader = CreateTransformedInterpolateData(
            baseShapesLoader,
            shapesInfo.PADDING,
            *shapesInfo.DEFAULT_JOINT_FACTORS,
            *shapesInfo.INTERPOLATE_JOINT_FACTORS
        )

    def loadInterpolationData(self):
        return self.__mnistLoader.loadInterpolationData()

    def dataPointShape(self):
        return shapesInfo.IMAGE_DIMENSIONS, shapesInfo.LABEL_DIMENSIONS
