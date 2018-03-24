import matplotlib
matplotlib.use('Agg')

from dataset.basicLoader.NorbLoader import NorbLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


norbLoader = ScaleBetweenZeroAndOne(NorbLoader("../res/norb"), 0, 255)
(xTrain, yTrain), (xVal, yVal), (xTest, yTest) = norbLoader.loadData()
xTrain = xTrain.reshape(xTrain.shape[0],xTrain.shape[1]*xTrain.shape[2])
print(xTrain.shape)

pca = PCA(n_components=1000)

start = time.time()
pca.fit(xTrain)
end = time.time()
elapsed = end - start
print("Time elapsed: {}".format(elapsed))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.xlim(0, 1.0)
plt.ylabel('Cumulative explained variance)')
plt.savefig('norb_pca_graph')
