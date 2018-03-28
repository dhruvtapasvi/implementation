import matplotlib.pyplot as plt
from skimage import data, color
import numpy as np
from skimage.transform import resize
from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne

norbLoader = ScaleBetweenZeroAndOne(NorbLoader("./res/norb"), 0, 255)
(xTrain, _), _, _ = norbLoader.loadData()

# Display logic
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

numImages = 20
images = xTrain[0:numImages]

im_resized = np.array([resize(image, (32, 32)) for image in images])

displayArraySamplesSideBySide([images, im_resized], numImages, "./out/downsample.png")