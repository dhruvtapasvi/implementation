import dataset.loaderPackaged as datasetPackages
import config.packagedConfigs as configs
from experiment.ExperimentalConfigTuple import ExperimentalConfigTuple
from evaluation.metric.SquaredError import SquaredError
from evaluation.metric.BinaryCrossEntropy import BinaryCrossEntropy


experimentalConfigTuples = [
    # MNIST
    # ExperimentalConfigTuple(datasetPackages.mnistPackage, configs.conv_28x28_3_8_ENC_1024_DEC_1024_LAT_32_bce, 500, 100, SquaredError(1), BinaryCrossEntropy()),
    # ExperimentalConfigTuple(datasetPackages.mnistPackage, configs.deepDense_28x28_ENC_512x2_1024_DEC_512x3_LAT_32_bce, 500, 200, SquaredError(1), BinaryCrossEntropy()),

    # MNIST transformed
    # ExperimentalConfigTuple(datasetPackages.mnistTransformedLimitedRotationPackage, configs.deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce, 500, 100, SquaredError(1), BinaryCrossEntropy()),
    # ExperimentalConfigTuple(datasetPackages.mnistTransformedLimitedRotationPackage, configs.conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce, 500, 100, SquaredError(1), BinaryCrossEntropy()),

    # Shapes
    ExperimentalConfigTuple(datasetPackages.shapesTransformedLimitedRotationPackage, configs.deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce, 500, 100, SquaredError(1), BinaryCrossEntropy()),
    # ExperimentalConfigTuple(datasetPackages.shapesTransformedLimitedRotationPackage, configs.conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce, 500, 100, SquaredError(1), BinaryCrossEntropy()),

    # NORB
    # ExperimentalConfigTuple(datasetPackages.norbPackage, configs.deepDense_96x96_ENC_1024_2048_2048_DEC_2048_2048_1024_LAT_32_bce, 500, 100, SquaredError(1), BinaryCrossEntropy()),
    # ExperimentalConfigTuple(datasetPackages.norbPackage, configs.conv_96x96_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce, 250, 100, SquaredError(1), BinaryCrossEntropy())
]
