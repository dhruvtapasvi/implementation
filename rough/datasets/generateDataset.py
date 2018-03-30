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

io.imshow(image)
io.show()

image = pad(image, 14, 'constant')

io.imshow(image)
io.show()

trans1 = tf.AffineTransform(translation=(-28, -28))
maintrans = tf.AffineTransform(scale=(math.sqrt(1/2),math.sqrt(1/2)))
trans2 = tf.AffineTransform(translation=(28, 28))
modified = tf.warp(image, (trans1 + (maintrans + trans2)).inverse)
io.imshow(modified)
io.show()

# # Draw a circle
# im = np.zeros((200, 200), dtype=np.float32)
# elx, ely = ellipse(100, 100, 50, 50)
# im[elx, ely] = 1
# io.imshow(im)
# io.show()
