import dataset.loaderPackaged as datasetPackages
import config.packagedConfigs as configs


experimentalDatasetModelPairs = [
    (datasetPackages.mnistPackage, configs.dense_28x28_keras_autoencoders_tutorial),
    (datasetPackages.mnistPackage, configs.conv_28x28_3_8_256_2_bce),

    (datasetPackages.mnistTransformedPackage, configs.conv_64x64_7_16_256_32_bce),

    (datasetPackages.shapesTransformedPackage, configs.conv_64x64_7_16_256_32_bce),

    (datasetPackages.norbPackage, configs.conv_96x96_6_16_256_10_bce)
]
