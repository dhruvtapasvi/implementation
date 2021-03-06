import numpy as np
import dataset.info.ShapesBaseInfo as shapesInfo

from dataset.loader.DatasetLoader import DatasetLoader


class ShapesBase(DatasetLoader):
    def loadData(self) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        shapeDraws = shapesInfo.SHAPE_DRAWS
        baseX = np.zeros((len(shapeDraws),) + shapesInfo.IMAGE_SIZE, dtype=np.uint8)
        for index, shapeDraw in enumerate(shapeDraws):
            shapeDraw(baseX[index])
        baseY = np.arange(len(shapeDraws)).reshape((len(shapeDraws), 1))
        train = np.array(baseX), np.array(baseY)
        validation = np.array(baseX), np.array(baseY)
        test = np.array(baseX), np.array(baseY)
        return train, validation, test

    def dataPointShape(self):
        return shapesInfo.IMAGE_SIZE, shapesInfo.LABEL_SIZE
