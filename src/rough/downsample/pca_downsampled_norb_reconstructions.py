import pickle
import matplotlib.pyplot as plt
import numpy as np
from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from dataset.preprocessLoader.Downsample import Downsample


norbLoader = Downsample(ScaleBetweenZeroAndOne(NorbLoader("./res/norb"), 0, 255), 32, 32)
(xTrain, yTrain), (xVal, yVal), (xTest, yTest) = norbLoader.loadData()

def plotPcaReconstructions(pca, numImages, images, figName):
    sel = images[0:numImages]
    selReshape = sel.reshape(numImages, 32 * 32)
    selPca = pca.transform(selReshape)
    selRecon = np.clip(pca.inverse_transform(selPca), 0.0, 0.999)
    selReconOrigShape = selRecon.reshape(sel.shape)

    plt.figure(figsize=(2, numImages))
    for i in range(numImages):
        num = plt.subplot(numImages, 2, 2 * i + 1)
        plt.imshow(sel[i])
        plt.gray()
        num.get_xaxis().set_visible(False)
        num.get_yaxis().set_visible(False)

        num = plt.subplot(numImages, 2, 2 * i + 2)
        plt.imshow(selReconOrigShape[i])
        plt.gray()
        num.get_xaxis().set_visible(False)
        num.get_yaxis().set_visible(False)

    plt.savefig("./pcaDownsample/" + figName + ".png")

pcaNComps = [ 100 ]
tvt = [(xTrain, "train"), (xVal, "val"), (xTest, "test")]
for i in pcaNComps:
    pca = pickle.load(open("./pcaDownsample/norb_pca_downsampled_" + str(i) + ".p", "rb"))
    for (set, name) in tvt:
        plotPcaReconstructions(pca, 20, set, "pca_recon_norb_" + str(i) + "_" + name)
