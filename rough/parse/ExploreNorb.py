from parse.NorbParser import NorbParser
from parse.NorbParsed import NorbParsed
import matplotlib.pyplot as plt
from datasets.NorbLoader import NorbLoader


numImages = 10
norbLoader = NorbLoader('../../res/norb')

(x_norbTrain, y_norbTrain), _ = norbLoader.loadData()
x_norbTrain = x_norbTrain[0:numImages]
y_norbTrain = y_norbTrain[0:numImages]

print(y_norbTrain)

plt.figure(figsize=(numImages, 1))
for i in range(numImages):
    image = (x_norbTrain[i, 0]).astype('float') / 255.
    num = plt.subplot(1, numImages, i + 1)
    plt.imshow(image)
    plt.gray()
    num.get_xaxis().set_visible(False)
    num.get_yaxis().set_visible(False)
plt.show()
