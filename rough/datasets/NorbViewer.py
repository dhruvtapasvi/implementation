from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from dataset.preprocessLoader.Downsample import Downsample
import matplotlib.pyplot as plt

def displayArraySamplesSideBySide(arrays, numSamplesToDisplay, fileName):
    numArrays = len(arrays)
    plt.figure(figsize=(numArrays, numSamplesToDisplay))
    for i in range(numSamplesToDisplay):
        for index, array in enumerate(arrays):
            num = plt.subplot(numSamplesToDisplay, numArrays, numArrays * i + index + 1)
            plt.imshow(array[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
    plt.savefig(fileName)

norbLoader = Downsample(ScaleBetweenZeroAndOne(NorbLoader("./res/norb"), 0, 255), 32, 32)
(xTrain, _), _, _ = norbLoader.loadData()

displayArraySamplesSideBySide([xTrain[0:20]], 20, "./out/downsampleloader.png")