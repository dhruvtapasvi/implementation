import matplotlib.pyplot as plt


def scatterPlotDisplay(points, labels, outFile):
    xPoints = points[:, 0]
    yPoints = points[:, 1]

    # TODO: Label axes and colour bar with appropriate titles

    plt.figure(figsize=(6,6))
    plt.scatter(xPoints, yPoints, c=labels)
    plt.colorbar()
    plt.savefig(outFile)
