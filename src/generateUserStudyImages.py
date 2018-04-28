from experiment.UserStudyGenerateImagesExperiment import UserStudyGenerateImagesExperiment
from experiment.BuildModelExperiment import BuildModelExperiment
from experiment.LoadModelExperiment import LoadModelExperiment
from config import packagedConfigs as configs
from dataset import loaderPackaged as loaders


NUMBER_USER_STUDIES = 20


DATASET_MODELS_PAIRINGS = [
    (
        loaders.mnistTransformedLimitedRotationPackage,
        [
            configs.deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce,
            configs.conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce
        ]
    ),
    (
        loaders.shapesTransformedLimitedRotationPackage,
        [
            configs.deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce,
            configs.conv_64x64_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce
        ]
    ),
    (
        loaders.norbPackage,
        [
            configs.deepDense_96x96_ENC_1024_2048_2048_DEC_2048_2048_1024_LAT_32_bce,
            configs.conv_96x96_6_16_ENC_1024x3_DEC_1024x3_LAT_32_bce
        ]
    )
]

for dataset, models in DATASET_MODELS_PAIRINGS:

    modelModelNamePairs = []
    for model in models:
        buildModel = BuildModelExperiment(model)
        variationalAutoencoder = buildModel.run()

        loadModelWeights = LoadModelExperiment(dataset.name + "_" + model.stringDescriptor, variationalAutoencoder)
        loadModelWeights.run()

        modelModelNamePairs.append((variationalAutoencoder, model.stringDescriptor))

    datasetSplits = dataset.datasetLoader.loadData()
    interpolationSubdatasets = dataset.interpolateLoader.loadInterpolationData()

    userStudyGenerate = UserStudyGenerateImagesExperiment(
        NUMBER_USER_STUDIES,
        datasetSplits,
        interpolationSubdatasets,
        dataset.name,
        modelModelNamePairs
    )
    userStudyGenerate.run()
