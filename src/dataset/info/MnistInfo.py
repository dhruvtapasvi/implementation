from enum import Enum


MNIST_IMAGE_DIMENSIONS = (28, 28)
MNIST_LABEL_DIMENSIONS = (1,)
MNIST_VALIDATION_SPLIT = 50000


class MnistLabelIndex(Enum):
    NUMBER = 0
