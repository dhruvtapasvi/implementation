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
    string = \
        "\\begin{subfigure}{\\linewidth}\n" +  \
            "\centering\n" + \
            "\\includegraphics[width=10cm]{vaeReconstructionResults/" + dataset + "_" + archDescription + "_" + reconstructionType + "Reconstructions.png}\n" + \
            "\\caption{" + datasetName + ", " + archName + ", " + reconstructionName + "}\n" + \
        "\\end{subfigure}\n"
    return string
    # return "\\subfloat[" \
    #     + datasetName \
    #     + ", " \
    #     + archName \
    #     + ", " \
    #     + reconstructionName \
    #     + "]{\\includegraphics[width=7cm]{vaeReconstructionResults/" \
    #     + dataset \
    #     + "_" \
    #     + archDescription \
    #     + "_" \
    #     + reconstructionType \
    #     + "Reconstructions.png}}\n"


def reconstructionFig():
    output = ""
    for dataset in datasetInfo.DATASET_ORDER:
        for architecture in datasetInfo.ARCH_TYPES:
            for index, reconstructionType in enumerate(RECONSTRUCTION_TYPES):
                output += reconstructionSubfig(dataset, architecture, reconstructionType)
                output += "\n"
    return output


def samplingSubfig(dataset, arch):
    datasetName = datasetInfo.DATASET_NAMES[dataset]
    archDescription = datasetInfo.DATASET_ARCH_NAMES[dataset][arch]
    archName = datasetInfo.ARCH_NAMES[arch]
    string = \
        "\\begin{subfigure}{\\linewidth}\n" +  \
            "\centering\n" + \
            "\\includegraphics[width=14cm]{vaeSamplingResults/" + dataset + "_" + archDescription + "_randomSampling.png}\n" + \
            "\\caption{" + datasetName + ", " + archName + "}\n" + \
        "\\end{subfigure}\n"
    return string
    # return "\\subfloat[" \
    #     + datasetName \
    #     + ", " \
    #     + archName \
    #     + "]{\\includegraphics[width=7cm]{vaeSamplingResults/" \
    #     + dataset \
    #     + "_" \
    #     + archDescription \
    #     + "_randomSampling.png}}\n"


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
            output += "\n"
    return output



print(reconstructionFig())
