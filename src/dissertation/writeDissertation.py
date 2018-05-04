from dissertation import datasetInfo


RECONSTRUCTION_TYPES = ["train", "test"]


RECONSTRUCTION_TYPE_NAMES = {
    "train": "Training",
    "test": "Testing"
}


def reconstructionSubfig(dataset, arch, reconstructionType):
    datasetName = datasetInfo.DATASET_NAMES[dataset]
    archDescription = datasetInfo.DATASET_ARCH_NAMES[dataset][arch]
    archName = datasetInfo.ARCH_NAMES[arch]
    reconstructionName = RECONSTRUCTION_TYPE_NAMES[reconstructionType]
    return "\\subfloat[" \
        + datasetName \
        + ", " \
        + archName \
        + ", " \
        + reconstructionName \
        + "]{\\includegraphics[width=7cm]{vaeReconstructionResults/" \
        + dataset \
        + "_" \
        + archDescription \
        + "_" \
        + reconstructionType \
        + "Reconstructions.png}}\n"


def reconstructionFig():
    output = ""
    for dataset in datasetInfo.DATASET_ORDER:
        for architecture in datasetInfo.ARCH_TYPES:
            for index, reconstructionType in enumerate(RECONSTRUCTION_TYPES):
                output += reconstructionSubfig(dataset, architecture, reconstructionType)
                output += "\n" if index == len(RECONSTRUCTION_TYPES) -1 else "\\hfill\n"
    return output


def samplingSubfig(dataset, arch):
    datasetName = datasetInfo.DATASET_NAMES[dataset]
    archDescription = datasetInfo.DATASET_ARCH_NAMES[dataset][arch]
    archName = datasetInfo.ARCH_NAMES[arch]
    return "\\subfloat[" \
        + datasetName \
        + ", " \
        + archName \
        + "]{\\includegraphics[width=7cm]{vaeSamplingResults/" \
        + dataset \
        + "_" \
        + archDescription \
        + "_randomSampling.png}}\n"


def interpolationSubfig(dataset, arch):
    datasetName = datasetInfo.DATASET_NAMES[dataset]
    archDescription = datasetInfo.DATASET_ARCH_NAMES[dataset][arch]
    archName = datasetInfo.ARCH_NAMES[arch]
    return "\\subfloat[" \
        + datasetName \
        + ", " \
        + archName \
        + "]{\\includegraphics[width=7cm]{vaeSamplingResults/" \
        + dataset \
        + "_" \
        + archDescription \
        + "_randomSampling.png}}\n"


def otherFig(filenameFunction):
    output = ""
    for dataset in datasetInfo.DATASET_ORDER:
        for index, architecture in enumerate(datasetInfo.ARCH_TYPES):
            output += filenameFunction(dataset, architecture)
            if index < len(datasetInfo.ARCH_TYPES) - 1:
                output += "\\hfill"
            output += "\n"
    return output



print(otherFig(samplingSubfig))
