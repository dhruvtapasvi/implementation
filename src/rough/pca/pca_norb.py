import matplotlib
matplotlib.use('Agg')

import pickle
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

pca = PCA(n_components=100)

start = time.time()
pca.fit(xTrain)
end = time.time()
elapsed = end - start
print("Time elapsed: {}".format(elapsed))

for i in range(100):
    print(i, pca.explained_variance_ratio_[i])

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylim(0, 1.0)
plt.ylabel('Cumulative explained variance)')
plt.savefig('norb_pca_graph')

pickle.dump(pca, open("./norb_pca_100.p", "wb"))