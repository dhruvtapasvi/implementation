import pickle
import matplotlib.pyplot as plt
import numpy as np
from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne

norbLoader = ScaleBetweenZeroAndOne(NorbLoader("./res/norb"), 0, 255)
(xTrain, yTrain), (xVal, yVal), (xTest, yTest) = norbLoader.loadData()

pca = pickle.load(open("./pca/norb_pca_100.p", "rb"))

def plotPcaReconstructions(pca, numImages, images, figName):
    sel = images[0:numImages]
    selReshape = sel.reshape(numImages, 96 * 96)
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

    plt.savefig("./out/" + figName + ".png")

pcaNComps = [ 100, 500, 1000, 2000 ]
tvt = [(xTrain, "train"), (xVal, "val"), (xTest, "test")]
for i in pcaNComps:
    pca = pickle.load(open("./pca/norb_pca_" + str(i) + ".p", "rb"))
    for (set, name) in tvt:
        plotPcaReconstructions(pca, 20, set, "pca_recon_norb_" + str(i) + "_" + name)
