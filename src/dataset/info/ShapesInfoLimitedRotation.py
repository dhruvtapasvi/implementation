import math


RANDOM_GENERATION_SEED = "GENERATE_SHAPES_DATASET"
RANGE = (0, 255)


IMAGE_DIMENSIONS = (64, 64)
PADDING = ((21, 21), (21, 21))
LABEL_DIMENSIONS = (6,)


HOME = "/shapes_LimitedRotation"


BASE_TRAIN_PROPORTION = 5
BASE_VALIDATION_PROPORTION = 1
BASE_TEST_PROPORTION = 1
BASE_NUM_SAMPLES = 4000


DEFAULT_ROTATION_NUM_DELTAS = 5
DEFAULT_SHEAR_NUM_DELTAS = 9
DEFAULT_ENLARGEMENT_NUM_DELTAS = 10


TRANSFORM_MIN_MAX_ROTATIONS = (-0.25 * math.pi, 0.25 * math.pi)
TRANSFORM_SHEAR_FACTOR = 0.45
TRANSFORM_LOG2_STRETCH_FACTOR = 0.5
DEFAULT_ROTATION_FACTOR = 0.0
DEFAULT_SHEAR_FACTOR = 0.0
DEFAULT_LOG2_STRETCH_FACTOR = 0.0
DEFAULT_JOINT_FACTORS = (DEFAULT_ROTATION_FACTOR, DEFAULT_SHEAR_FACTOR, DEFAULT_LOG2_STRETCH_FACTOR)

INTERPOLATE_ROTATION_FACTOR = 0.2 * math.pi
INTERPOLATE_ROTATION_FACTORS = (-INTERPOLATE_ROTATION_FACTOR, INTERPOLATE_ROTATION_FACTOR, 0.0, 0.75 * math.pi)
INTERPOLATE_SHEAR_FACTOR = 0.3
INTERPOLATE_SHEAR_FACTORS = (-INTERPOLATE_SHEAR_FACTOR, INTERPOLATE_SHEAR_FACTOR, 0.0, 1.0)
LOG2_INTERPOLATE_STRETCH_FACTOR = 0.3
INTERPOLATE_LOG2_STRETCH_FACTORS = (-LOG2_INTERPOLATE_STRETCH_FACTOR, LOG2_INTERPOLATE_STRETCH_FACTOR, 0.0, 1.0)
INTERPOLATE_JOINT_FACTORS = (INTERPOLATE_ROTATION_FACTORS, INTERPOLATE_SHEAR_FACTORS, INTERPOLATE_LOG2_STRETCH_FACTORS)
