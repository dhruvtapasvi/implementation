DATASET_ORDER = [
    "mnist",
    "mnistTransformedLimitedRotation",
    "shapesTransformedLimitedRotation",
    "norb"
]


INTERPOLATION_DATASET_ORDER = [
    "mnistTransformedLimitedRotation",
    "shapesTransformedLimitedRotation",
    "norb"
]


DATASET_NAMES = {
    "mnistTransformedLimitedRotation": "Transformed MNIST",
    "mnist": "MNIST",
    "shapesTransformedLimitedRotation": "Shapes",
    "norb": "NORB"
}


ARCH_TYPES = [
    "dense",
    "conv"
]


DATASET_ARCH_NAMES = {
    "mnistTransformedLimitedRotation": {
        "dense": "deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce",
        "conv": "conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce"
    }, "mnist": {
        "dense": "deepDense_28x28_ENC_512x2_1024_DEC_512x3_LAT_32_bce",
        "conv": "conv_28x28_3_8_ENC_1024_DEC_1024_LAT_32_bce"
    }, "shapesTransformedLimitedRotation": {
        "dense": "deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce",
        "conv": "conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce"
    }, "norb": {
        "dense": "deepDense_96x96_ENC_1024_2048_2048_DEC_2048_2048_1024_LAT_32_bce",
        "conv": "conv_96x96_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce"
    }
}


LOSSES_CATEGORIES = [
    "kl",
    "reconstruction",
    "total"
]


INTERPOLATION_TECHNIQUES = [
    "interpolateLatentSpace",
    "interpolateImageSpace",
    "left",
    "right",
    "outside",
    "randomImage"
]


INTERPOLATE_TECHNIQUE_NAMES = {
    "interpolateLatentSpace": "Interpolate\nin the\nLatent\nSpace",
    "interpolateImageSpace": "Interpolate\nin the\nImage\nSpace",
    "left": "First\nimage",
    "right": "Second\nimage",
    "outside": "An image\nlying\noutside\ninterpolation\nrange",
    "randomImage": "A random\nimage"
}

