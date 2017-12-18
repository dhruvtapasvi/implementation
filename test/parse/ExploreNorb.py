from parse.NorbParser import NorbParser
import matplotlib.pyplot as plt

numImages = 10

# Print the first few images of NORB and their categories and image details:

with open('../../res/norb/norb_train_image', 'r') as f:
    print("IMAGE:")
    norbParser = NorbParser(f)
    result = norbParser.parse()
    print(result.type)
    print(result.numDimensions)
    print(result.dimensions)
    print(result.data.shape)

    plt.figure(figsize=(10, 1))
    for i in range(numImages):
        image = (result.data[i, 0]).astype('float') / 255.
        num = plt.subplot(1, 10, i+1)
        plt.imshow(image)
        plt.gray()
        num.get_xaxis().set_visible(False)
        num.get_yaxis().set_visible(False)

    plt.show()

with open('../../res/norb/norb_train_category', 'r') as g:
    print("CATEGORY:")
    norbParser = NorbParser(g)
    result = norbParser.parse()
    print(result.type)
    print(result.numDimensions)
    print(result.dimensions)
    print(result.data.shape)
    print(result.data[0:numImages])

with open('../../res/norb/norb_train_info', 'r') as h:
    print("DETAILS:")
    norbParser = NorbParser(h)
    result = norbParser.parse()
    print(result.type)
    print(result.numDimensions)
    print(result.dimensions)
    print(result.data.shape)
    print(result.data[0:numImages])
