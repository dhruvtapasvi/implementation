import numpy as np
import matplotlib.pyplot as plt


from evaluation.results import packageResults
from dissertation import datasetInfo
from config.routes import getRecordedResultsRoute


NUMBER_FORMAT = "{:.1f}"


interpolationResults = packageResults.interpolationResults.getDictionary()


# Distance in image space:
for dataset in datasetInfo.INTERPOLATION_DATASET_ORDER:
    interpolationFactors = list(interpolationResults[dataset][datasetInfo.DATASET_ARCH_NAMES[dataset]["dense"]].keys())
    for interpolationFactor in interpolationFactors:
        metricResults = [
            interpolationResults[dataset][datasetInfo.DATASET_ARCH_NAMES[dataset]["conv"]][interpolationFactor]["interpolateLatentSpace"]["metricImageSpace"]
        ] + [
            interpolationResults[dataset][datasetInfo.DATASET_ARCH_NAMES[dataset]["dense"]][interpolationFactor][interpolationTechnique]["metricImageSpace"]
            for interpolationTechnique in datasetInfo.INTERPOLATION_TECHNIQUES
        ]
        x = np.arange(len(datasetInfo.INTERPOLATION_TECHNIQUES) + 1)
        means = np.array(list(map(lambda x: x.mean, metricResults)))
        stds = np.array(list(map(lambda x: x.standardDeviation, metricResults)))
        labels = [datasetInfo.INTERPOLATE_TECHNIQUE_NAMES["interpolateLatentSpace"] + "\n(convolutional)"] + \
                 [datasetInfo.INTERPOLATE_TECHNIQUE_NAMES[interpolationTechnique] + ("\n(dense)" if interpolationTechnique == "interpolateLatentSpace" else "") for interpolationTechnique in datasetInfo.INTERPOLATION_TECHNIQUES]

        plt.figure(figsize=(8, 6))

        bars = plt.bar(x, means, yerr=stds, capsize=5)

        plt.xticks(x, labels)
        plt.xlabel("Interpolation Technique")

        plt.ylabel("Distance in Image Space (Binary Cross-entropy)")
        plt.ylim(ymin=0)

        maxVal = max(map(lambda x: x.mean + x.standardDeviation, metricResults))
        extraHeight = 0.0125 * maxVal
        for bar, error, mean in zip(bars, stds, means):
            plt.text(
                bar.get_x() + 0.5 * bar.get_width(),
                mean + error + extraHeight,
                NUMBER_FORMAT.format(mean),
                ha="center",
                fontweight="bold",
                fontsize=12
            )

        plt.tight_layout()

        plt.savefig(getRecordedResultsRoute(dataset + "_" + interpolationFactor + "_" + "metricImageSpace.png"))
        plt.close()


# Distance in latent space:
for dataset in datasetInfo.INTERPOLATION_DATASET_ORDER:
    interpolationFactors = list(interpolationResults[dataset][datasetInfo.DATASET_ARCH_NAMES[dataset]["dense"]].keys())
    for interpolationFactor in interpolationFactors:
        for architecture in datasetInfo.ARCH_TYPES:
            metricResults = [
                interpolationResults[dataset][datasetInfo.DATASET_ARCH_NAMES[dataset][architecture]][interpolationFactor][interpolationTechnique]["metricLatentSpace"]
                for interpolationTechnique in datasetInfo.INTERPOLATION_TECHNIQUES
            ]
            x = np.arange(len(datasetInfo.INTERPOLATION_TECHNIQUES))
            means = np.array(list(map(lambda x: x.mean, metricResults)))
            stds = np.array(list(map(lambda x: x.standardDeviation, metricResults)))
            labels = [datasetInfo.INTERPOLATE_TECHNIQUE_NAMES[interpolationTechnique] for interpolationTechnique in datasetInfo.INTERPOLATION_TECHNIQUES]

            plt.figure(figsize=(8, 6))

            bars = plt.bar(x, means, yerr=stds, capsize=5)

            plt.xticks(x, labels)
            plt.xlabel("Interpolation Technique")

            plt.ylabel("Distance in Latent Space")
            plt.ylim(ymin=0)

            maxVal = max(map(lambda x: x.mean + x.standardDeviation, metricResults))
            extraHeight = 0.0125 * maxVal
            for bar, error, mean in zip(bars, stds, means):
                plt.text(
                    bar.get_x() + 0.5 * bar.get_width(),
                    mean + error + extraHeight,
                    NUMBER_FORMAT.format(mean),
                    ha="center",
                    fontweight="bold",
                    fontsize=12
                )

            plt.tight_layout()

            plt.savefig(getRecordedResultsRoute(dataset + "_" + architecture + "_" + interpolationFactor + "_" + "metricLatentSpace.png"))
            plt.close()
