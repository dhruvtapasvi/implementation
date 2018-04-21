import dataset.loaderPackaged as datasetPackages
import config.packagedConfigs as configs
from experiment.ExperimentalConfigTuple import ExperimentalConfigTuple


experimentalConfigTuples = [
    # MNIST
    ExperimentalConfigTuple(datasetPackages.mnistPackage, configs.conv_28x28_3_8_ENC_1024_DEC_1024_LAT_32_bce, 500, 100),
    ExperimentalConfigTuple(datasetPackages.mnistPackage, configs.deepDense_28x28_ENC_512x2_1024_DEC_512x3_LAT_32_bce, 500, 200),

    # MNIST Transformed


    ExperimentalConfigTuple(datasetPackages.mnistTransformedPackage, configs.conv_64x64_7_16_256_32_bce, 1000, 100),

    ExperimentalConfigTuple(datasetPackages.shapesTransformedPackage, configs.conv_64x64_7_16_256_32_bce, 1000, 100),

    ExperimentalConfigTuple(datasetPackages.norbPackage, configs.conv_96x96_6_16_256_10_bce, 1000, 100)
]
