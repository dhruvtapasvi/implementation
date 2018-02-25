import matplotlib.pyplot as plt

from datasets.basicLoaders.MnistLoader import MnistLoader

mnistLoader = MnistLoader()
(_, _), (xTest, yTest) = mnistLoader.loadData()
xTest = xTest.astype('float') / 255.

numToSeeSqrt = 20
print(yTest[0:numToSeeSqrt])

numToSee = numToSeeSqrt * numToSeeSqrt
plt.figure(figsize=(numToSeeSqrt * 1.2, numToSeeSqrt * 1.2))
for i in range(numToSee):
    num = plt.subplot(numToSeeSqrt, numToSeeSqrt, i + 1)
    plt.imshow(xTest[i])
    plt.gray()
    num.get_xaxis().set_visible(False)
    num.get_yaxis().set_visible(False)
    num.set_title(str(i))
plt.savefig('mnistnumbers.png')

