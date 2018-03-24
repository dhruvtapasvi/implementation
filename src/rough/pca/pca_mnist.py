import matplotlib
matplotlib.use('Agg')

import pickle
import time
from dataset.basicLoader.MnistLoader import MnistLoader
from dataset.preprocessLoader.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


mnistLoader = ScaleBetweenZeroAndOne(MnistLoader(), 0, 255)
(xTrain, yTrain), (xVal, yVal), (xTest, yTest) = mnistLoader.loadData()
xTrain = xTrain.reshape(xTrain.shape[0],xTrain.shape[1]*xTrain.shape[2])
print(xTrain.shape)

pca = PCA(n_components=100)

start = time.time()
pca.fit(xTrain)
end = time.time()
elapsed = end-start
print("Time taken: {}".format(elapsed))

for i in range(100):
    print(i, pca.explained_variance_ratio_[i])

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance)')
plt.savefig('mnist_pca_graph.png')

pickle.dump(pca, open("./mnist_pca_100.p", "wb"))
