from enum import Enum


MNIST_RANGE = (0, 255)
MNIST_IMAGE_DIMENSIONS = (28, 28)
MNIST_LABEL_DIMENSIONS = (1,)
MNIST_VALIDATION_SPLIT = 50000


MNIST_INTERPOLATION_INSTANCES = [
    # Format is (left, right, centre [optional], outside [optional])
    (5, 191),  # Right slanted vs left slanted 1
    (0, 243),  # 7 without horizontal bar vs with horiztonal bar
    (148, 311),  # Thin vs fat 0
    (59, 207),  # Uncurved vs very curved belly of 5
    (140, 259),  # Differing 6 styles
    (99, 17),  # 9 to 7
    (270, 52)  # 3 to 5
]


class MnistLabelIndex(Enum):
    NUMBER = 0
