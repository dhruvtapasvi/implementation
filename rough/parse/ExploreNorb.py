from parse.NorbParser import NorbParser
from parse.NorbParsed import NorbParsed
import matplotlib.pyplot as plt


def parseAndPrintSummary(fileHandle) -> NorbParsed:
    result = norbParser.parse(fileHandle)
    print(result.underlyingType)
    print(result.numDimensions)
    print(result.dimensions)
    print(result.data.shape)
    return result


numImages = 10
norbParser = NorbParser()

# Print the first numImages images of NORB and their categories and image details:

with open('../../res/norb/norb_train_category', 'r') as categoryFile:
    print("CATEGORY:")
    result = parseAndPrintSummary(categoryFile)
    print(result.data[0:numImages])

with open('../../res/norb/norb_train_info', 'r') as detailsFile:
    print("DETAILS:")
    result = parseAndPrintSummary(detailsFile)
    print(result.data[0:numImages])

with open('../../res/norb/norb_train_image', 'r') as imageFile:
    print("IMAGE:")
    result = parseAndPrintSummary(imageFile)

    plt.figure(figsize=(numImages, 1))
    for i in range(numImages):
        image = (result.data[i, 0]).astype('float') / 255.
        num = plt.subplot(1, numImages, i+1)
        plt.imshow(image)
        plt.gray()
        num.get_xaxis().set_visible(False)
        num.get_yaxis().set_visible(False)

    plt.show()
