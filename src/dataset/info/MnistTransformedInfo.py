from enum import Enum


RANDOM_GENERATION_SEED = "GENERATE_MNIST_TRANSFORMED"
IMAGE_DIMENSIONS = (64, 64)
LABEL_DIMENSIONS = (6,)


class LabelIndex(Enum):
    MNIST_INSTANCE_NUMBER = 0  # type is float, unfortunately
    NUMBER = 1  # type is float, unfortunately
    ROTATION_ANGLE = 2
    SHEAR_FACTOR = 3
    LOG2_STRETCH_FIRST_AXIS = 4
    LOG2_STRETCH_SECOND_AXIS = 5
