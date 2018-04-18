from skimage import draw
import numpy as np
import math


BASE_IMAGE_SIZE = (22, 22)
BASE_LABEL_SIZE = (1,)
RANDOM_GENERATION_SEED = "GENERATE_SHAPES_DATASET"
RANGE = (0, 255)


IMAGE_DIMENSIONS = (64, 64)
PADDING = ((21, 21), (21, 21))
LABEL_DIMENSIONS = (6,)


HOME = "/shapes"


BASE_TRAIN_PROPORTION = 5
BASE_VALIDATION_PROPORTION = 1
BASE_TEST_PROPORTION = 1
BASE_NUM_SAMPLES = 4000


TRANSFORM_SHEAR_FACTOR = 0.45
TRANSFORM_LOG2_STRETCH_FACTOR = 0.5
DEFAULT_ROTATION_FACTOR = 0.0
DEFAULT_SHEAR_FACTOR = 0.0
DEFAULT_LOG2_STRETCH_FACTOR = 0.0
DEFAULT_JOINT_FACTORS = (DEFAULT_ROTATION_FACTOR, DEFAULT_SHEAR_FACTOR, DEFAULT_LOG2_STRETCH_FACTOR)
INTERPOLATE_ROTATION_FACTORS = (-0.25 * math.pi, 0.25 * math.pi, 0.0, 0.75 * math.pi)
INTERPOLATE_SHEAR_FACTORS = (-TRANSFORM_SHEAR_FACTOR, TRANSFORM_SHEAR_FACTOR, 0.0, TRANSFORM_SHEAR_FACTOR + 0.5)
INTERPOLATE_LOG2_STRETCH_FACTORS = (-TRANSFORM_LOG2_STRETCH_FACTOR, TRANSFORM_LOG2_STRETCH_FACTOR, 0.0, TRANSFORM_LOG2_STRETCH_FACTOR + 0.5)
INTERPOLATE_JOINT_FACTORS = (INTERPOLATE_ROTATION_FACTORS, INTERPOLATE_SHEAR_FACTORS, INTERPOLATE_LOG2_STRETCH_FACTORS)


def drawRectangle(array: np.ndarray, rBounds, cBounds, fillValue=255):
    rLow, rHigh = rBounds
    cLow, cHigh = cBounds
    rectangle = draw.polygon([rLow, rLow, rHigh, rHigh], [cLow, cHigh, cHigh, cLow])
    array[rectangle] = fillValue


def drawTriangle(array: np.ndarray, vertices):
    (r1, c1), (r2, c2), (r3, c3) = vertices
    triangle = draw.polygon([r1, r2, r3], [c1, c2, c3])
    array[triangle] = 255


def drawCircle(array: np.ndarray, r, c, radius, truncBoundingBox=None):
    rrc, ccc = draw.circle(r, c, radius)
    array[rrc, ccc] = 255
    if truncBoundingBox is not None:
        (rLow, rHigh), (cLow, cHigh) = truncBoundingBox
        drawRectangle(array, (0, rLow), (0, 22), fillValue=0)
        drawRectangle(array, (rHigh, 22), (0, 22), fillValue=0)
        drawRectangle(array, (0, 22), (0, cLow), fillValue=0)
        drawRectangle(array, (0, 22), (cHigh, 22), fillValue=0)


def drawFirstShape(array: np.ndarray):
    # House shape
    drawRectangle(array, (11, 22), (0, 22))
    drawTriangle(array, ((11, 0), (0, 11), (11, 22)))


def drawSecondShape(array:np.ndarray):
    # L shape
    drawRectangle(array, (0, 15), (0, 7))
    drawRectangle(array, (15, 22), (0, 22))


def drawThirdShape(array: np.ndarray):
    # Arrow shape
    drawTriangle(array, ((15, 0), (0, 11), (15, 22)))
    drawRectangle(array, (15, 22), (7, 15))


def drawFourthShape(array: np.ndarray):
    # T shape
    drawRectangle(array, (0, 8), (0, 22))
    drawRectangle(array, (8, 22), (7, 15))


def drawFifthShape(array: np.ndarray):
    # Trapezium
    drawTriangle(array, ((22, 0), (0, 7), (22, 7)))
    drawRectangle(array, (0, 22), (7, 15))
    drawTriangle(array, ((22, 22), (0, 15), (22, 15)))


def drawSixthShape(array: np.ndarray):
    # 2 tooth comb
    drawRectangle(array, (0, 8), (0, 22))
    drawRectangle(array, (8, 22), (0, 7))
    drawRectangle(array, (8, 22), (15, 22))


def drawSeventhShape(array: np.ndarray):
    # Mushroom
    drawCircle(array, 11, 11, 11, ((0, 11), (0, 22)))
    drawRectangle(array, (11, 22), (7, 15))


def drawEighthShape(array: np.ndarray):
    # Weird T
    drawRectangle(array, (0, 15), (0, 22))
    drawRectangle(array, (15, 22), (7, 15))


SHAPE_DRAWS = [
    drawFirstShape,
    drawSecondShape,
    drawThirdShape,
    drawFourthShape,
    drawFifthShape,
    drawSixthShape,
    drawSeventhShape,
    drawEighthShape
]
