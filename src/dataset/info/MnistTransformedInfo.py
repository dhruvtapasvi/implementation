from enum import Enum
import math


RANDOM_GENERATION_SEED = "GENERATE_MNIST_TRANSFORMED"
IMAGE_DIMENSIONS = (64, 64)
LABEL_DIMENSIONS = (6,)


TRANSFORM_SHEAR_FACTOR = 0.14
INTERPOLATE_INCORRECT_SHEAR_FACTOR = 0.5
TRANSFORM_LOG2_STRETCH_FACTOR = 1.0 / 2.0
INTERPOLATE_INCORRECT_LOG_2_STRETCH_FACTOR = 1.0
INTERPOLATE_ROTATION_FACTORS = (-0.25 * math.pi, 0.25 * math.pi, 0.0, 0.75 * math.pi)


class LabelIndex(Enum):
    MNIST_INSTANCE_NUMBER = 0  # type is float, unfortunately
    NUMBER = 1  # type is float, unfortunately
    ROTATION_ANGLE = 2
    SHEAR_FACTOR = 3
    LOG2_STRETCH_FIRST_AXIS = 4
    LOG2_STRETCH_SECOND_AXIS = 5
