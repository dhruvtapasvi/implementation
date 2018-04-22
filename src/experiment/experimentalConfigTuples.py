import dataset.loaderPackaged as datasetPackages
import config.packagedConfigs as configs
from experiment.ExperimentalConfigTuple import ExperimentalConfigTuple
from evaluation.metric.SquaredError import SquaredError


experimentalConfigTuples = [
    ExperimentalConfigTuple(datasetPackages.mnistPackage, configs.dense_28x28_keras_autoencoders_tutorial, 2000, 50, SquaredError(1), SquaredError(1)),
    ExperimentalConfigTuple(datasetPackages.mnistPackage, configs.conv_28x28_3_8_256_2_bce, 2000, 50, SquaredError(1), SquaredError(1)),

    ExperimentalConfigTuple(datasetPackages.mnistTransformedPackage, configs.conv_64x64_7_16_256_32_bce, 1000, 100, SquaredError(1), SquaredError(1)),

    ExperimentalConfigTuple(datasetPackages.shapesTransformedPackage, configs.conv_64x64_7_16_256_32_bce, 1000, 100, SquaredError(1), SquaredError(1)),

    ExperimentalConfigTuple(datasetPackages.norbPackage, configs.conv_96x96_6_16_256_10_bce, 1000, 100, SquaredError(1), SquaredError(1))
]
