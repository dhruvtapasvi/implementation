from evaluation.results import packageResults
from dissertation import datasetInfo


NUMBER_FORMAT = "{:.1f}"


modelLosses = packageResults.modelLossResults.getDictionary()


def printLossesTable(lossType):
    output = ""
    for dataset in datasetInfo.DATASET_ORDER:
        output += datasetInfo.DATASET_NAMES[dataset] + " "
        for archType in datasetInfo.ARCH_TYPES:
            for lossCategory in datasetInfo.LOSSES_CATEGORIES:
                output += "& " + NUMBER_FORMAT.format(modelLosses[dataset][datasetInfo.DATASET_ARCH_NAMES[dataset][archType]][lossType][lossCategory]) + " "

        output += "\\\\\n"

    print(output)

printLossesTable("train")
printLossesTable("val")
printLossesTable("test")
