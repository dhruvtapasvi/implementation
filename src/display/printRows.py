import matplotlib.pyplot as plt


def printRows(arrays, numElementsToPrint, fileStem):
    """
    Display each row (singular index cutting across each array in arrays) as a single image
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
        plt.close()