import matplotlib.pyplot as plt

from dataset.basicLoader.NorbLoader import NorbLoader

numImages = 10
norbLoader = NorbLoader('../../res/norb')

(XTrain, YTrain), (XValidation, YValidation), (XTest, YTest) = norbLoader.loadData()
print(XTrain.shape, YTrain.shape, XValidation.shape, YValidation.shape, XTest.shape, YTest.shape)

y_norbTrain_trunc = YTrain[0:numImages]

print(y_norbTrain_trunc)

plt.figure(figsize=(numImages, 1))
for i in range(numImages):
    image = (XTrain[i]).astype('float') / 255.
    num = plt.subplot(1, numImages, i + 1)
    plt.imshow(image)
    plt.gray()
    num.get_xaxis().set_visible(False)
    num.get_yaxis().set_visible(False)
plt.show()
