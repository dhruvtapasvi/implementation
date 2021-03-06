from enum import Enum
import math
from config import routes


RANGE = (0, 255)
RANDOM_GENERATION_SEED = "GENERATE_MNIST_TRANSFORMED_LIMITED_ROTATION"
IMAGE_DIMENSIONS = (64, 64)
LABEL_DIMENSIONS = (6,)


PADDING = ((18, 18), (18, 18))
TRANSFORM_MIN_MAX_ROTATIONS = (-0.25 * math.pi, 0.25 * math.pi)
TRANSFORM_SHEAR_FACTOR = 0.14
TRANSFORM_LOG2_STRETCH_FACTOR = 1.0 / 2.0
DEFAULT_SHEAR_FACTOR = 0.0
DEFAULT_LOG2_STRETCH_FACTOR = 0.0
DEFAULT_ROTATION_FACTOR = 0.0
DEFAULT_JOINT_FACTORS = (DEFAULT_ROTATION_FACTOR, DEFAULT_SHEAR_FACTOR, DEFAULT_LOG2_STRETCH_FACTOR)

INTERPOLATE_SHEAR_FACTOR = 0.1
INTERPOLATE_SHEAR_FACTORS = (-INTERPOLATE_SHEAR_FACTOR, INTERPOLATE_SHEAR_FACTOR, 0.0, 0.5)
LOG2_INTERPOLATE_STRETCH_FACTOR = 0.3
INTERPOLATE_STRETCH_FACTORS = (-LOG2_INTERPOLATE_STRETCH_FACTOR, LOG2_INTERPOLATE_STRETCH_FACTOR, 0.0, 1.0)
INTERPOLATE_ROTATION_FACTOR = 0.2 * math.pi
INTERPOLATE_ROTATION_FACTORS = (-INTERPOLATE_ROTATION_FACTOR, INTERPOLATE_ROTATION_FACTOR, 0.0, 0.75 * math.pi)
INTERPOLATE_JOINT_FACTORS = (INTERPOLATE_ROTATION_FACTORS, INTERPOLATE_SHEAR_FACTORS, INTERPOLATE_STRETCH_FACTORS)


class LabelIndex(Enum):
    MNIST_INSTANCE_NUMBER = 0  # type is float, unfortunately
    NUMBER = 1  # type is float, unfortunately
    ROTATION_ANGLE = 2
    SHEAR_FACTOR = 3
    LOG2_STRETCH_FIRST_AXIS = 4
    LOG2_STRETCH_SECOND_AXIS = 5


HOME_STEM = routes.RESOURCE_ROUTE + "/mnistTransformed_LimitedRotations_"
HOME_1 = HOME_STEM + "1"
HOME_2 = HOME_STEM + "2"
HOME_5 = HOME_STEM + "5"
HOME_10 = HOME_STEM + "10"
