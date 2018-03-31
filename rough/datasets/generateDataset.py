from keras.datasets import mnist
from skimage import transform as tf
from skimage import io
from skimage.util import pad
from skimage.draw import ellipse
import math
import numpy as np

# Rotate the first test image from MNIST:
(xTrain, _), _ = mnist.load_data()
image = xTrain[0]

print(image)

io.imshow(image)
io.show()

image = pad(image, 14, 'constant')

io.imshow(image)
io.show()

print(np.amax(image))

trans1 = tf.AffineTransform(translation=(-28, -28))
maintrans = tf.AffineTransform(rotation=(3 * math.pi / 2))
trans2 = tf.AffineTransform(translation=(28, 28))
modified = tf.warp(image, (trans1 + (maintrans + trans2)).inverse)
io.imshow(modified)
io.show()

print(np.amax(modified))
print(modified.dtype)