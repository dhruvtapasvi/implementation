import matplotlib.pyplot as plt


def imagesArrayComparisonDisplay(arrays, fileName, startIndex=0, endIndex=None):
    """
    Display sample images from arrays side-by-side in a matrix format
    Each array is a column
    Each index is a row
    """
    numArrays = len(arrays)
    numElementsToDisplay = (min(map(len, arrays)) if endIndex is None else endIndex) - startIndex
    plt.figure(figsize=(numArrays, numElementsToDisplay))
    for i in range(startIndex, startIndex + numElementsToDisplay):
        for index, array in enumerate(arrays):
            num = plt.subplot(numElementsToDisplay, numArrays, numArrays * i + index + 1)
            plt.imshow(array[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
    plt.savefig(fileName)
