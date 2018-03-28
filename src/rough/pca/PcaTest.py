import pickle
import matplotlib.pyplot as plt
import numpy as np
from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne

norbLoader = ScaleBetweenZeroAndOne(NorbLoader("./res/norb"), 0, 255)
(xTrain, yTrain), (xVal, yVal), (xTest, yTest) = norbLoader.loadData()

pca = pickle.load(open("./pca/norb_pca_500.p", "rb"))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylim(0, 1.0)
plt.ylabel('Cumulative explained variance)')
plt.savefig('norb_pca_graph')