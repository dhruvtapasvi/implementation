import csv
import pickle
from config.routes import getUserStudyRoute, getUserStudyResponsesRoute, getRecordedResultsRoute
from dissertation import datasetInfo
import os
from typing import List
from evaluation.metric.MetricResult import MetricResult
import numpy as np
import matplotlib.pyplot as plt


NUMBER_FORMAT = "{:.1f}"


TASKS = ["interpolation", "reconstruction"]


TASKS_X_AXIS_TITLE = {
    "interpolation": "Interpolation Technique",
    "reconstruction": "Reconstruction Technique"
}


def getTaskTechniqueOrder(dataset: str, task:str):
    convSuffix = datasetInfo.DATASET_ARCH_NAMES[dataset]["conv"]
    denseSuffix = datasetInfo.DATASET_ARCH_NAMES[dataset]["dense"]
    if task == "interpolation":
        return [
            ("positiveControl", "Ideal\nresult"),
            ("interpolateLatentSpace_" + convSuffix, "Interpolate\nin the\nLatent\nSpace (conv)"),
            ("interpolateLatentSpace_" + denseSuffix, "Interpolate\nin the\nLatent\nSpace (dense)"),
            ("interpolateImageSpace", "Interpolate\nin the\nImage\nSpace"),
            ("left", "First\nimage\n(left)"),
            ("randomImage", "A random\nimage\n(random)")
        ]
    else:
        convSuffix = datasetInfo.DATASET_ARCH_NAMES[dataset]["conv"]
        denseSuffix = datasetInfo.DATASET_ARCH_NAMES[dataset]["dense"]
        return [
            ("positiveControl", "Ideal\nresult"),
            (convSuffix, "Conv\narchitecture"),
            (denseSuffix, "Dense\narchitecture"),
            ("randomImage", "A random\nimage\n(random)")
        ]



class UserStudyResponse:
    def __init__(self, questionInfo: str, responseInfo: str):
        questionInfoSplit = questionInfo.split("_")
        self.__dataset = questionInfoSplit[0]
        self.__task = questionInfoSplit[1]
        self.__instance = int(questionInfoSplit[2])
        responseInfoSplit = responseInfo.split(":")
        self.__score = int(responseInfoSplit[0])

    @property
    def dataset(self) -> str:
        return self.__dataset

    @property
    def task(self) -> str:
        return self.__task

    @property
    def instance(self) -> int:
        return self.__instance

    @property
    def score(self) -> int:
        return self.__score


recordedUserStudyResponses = []
for fileName in os.listdir(getUserStudyResponsesRoute("")):
    if fileName.endswith(".csv"):
        with open(getUserStudyResponsesRoute(fileName)) as csvFile:
            reader = csv.reader(csvFile, delimiter=',', quotechar='"')
            rows = [row[2:] for row in reader]
            responses = list(zip(*rows))
            recordedUserStudyResponses += [UserStudyResponse(*response) for response in responses]


def filterUserStudyResponses(userStudyResponses: List[UserStudyResponse], dataset: str, task: str) -> List[UserStudyResponse]:
    return [userStudyResponse for userStudyResponse in userStudyResponses if userStudyResponse.dataset == dataset and userStudyResponse.task == task]


def plotBarChart(metricResultsByTechnique, dataset: str, task: str):
    techniques, techniqueLabels = tuple(zip(*getTaskTechniqueOrder(dataset, task)))
    x = np.arange(len(techniques))
    metricResults = [(metricResultsByTechnique[technique] if technique in metricResultsByTechnique else MetricResult(np.array([0, 0]))) for technique in techniques]
    means = np.array(list(map(lambda x: x.mean, metricResults)))
    stds = np.array([metricResult.standardDeviation for metricResult in metricResults])

    # stds = np.array(list(map(lambda x: x.standardDeviation, metricResults)))
    plt.figure(figsize=(8, 6))

    plt.bar(x, means, yerr=stds, capsize=5)

    plt.xticks(x, techniqueLabels)
    plt.xlabel("Interpolation Technique")

    plt.ylabel("User Rating")
    plt.ylim(ymin=0)

    plt.tight_layout()

    plt.savefig(getRecordedResultsRoute("dataset_" + dataset + "_" + task + "_userRatings.png"))
    plt.close()


for dataset in datasetInfo.INTERPOLATION_DATASET_ORDER:
    for task in TASKS:
        filteredUserStudyResponses = filterUserStudyResponses(recordedUserStudyResponses, dataset, task)
        scoresByTechnique = {}
        with open(getUserStudyRoute(dataset + "/" + task + "/labels.p"), "rb") as labelFile:
            labels = pickle.load(labelFile)
            for filteredUserStudyResponse in filteredUserStudyResponses:
                scoreByTechnique = scoresByTechnique.setdefault(labels[filteredUserStudyResponse.instance], [])
                scoreByTechnique.append(filteredUserStudyResponse.score)
        metricsByTechnique = { technique: MetricResult(np.array(scores)) for technique, scores in scoresByTechnique.items()}
        plotBarChart(metricsByTechnique, dataset, task)
