import matplotlib.pyplot as plt


def printRows(arrays, numElementsToPrint, fileStem):
    """
    Display sample images from arrays side-by-side in a matrix format
    Each array is a column
    Each index is a row
    """
    numArrays = len(arrays)
    for i in range(0, numElementsToPrint):
        plt.figure(figsize=(numArrays, 1))
        for index, array in enumerate(arrays):
            num = plt.subplot(1, 3, index+1)
            plt.imshow(array[i])
            plt.gray()
            num.get_xaxis().set_visible(False)
            num.get_yaxis().set_visible(False)
        plt.savefig(fileStem + str(i), bbox_inches='tight', pad_inches=0)
