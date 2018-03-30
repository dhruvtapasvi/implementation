from keras.datasets import mnist
from skimage import transform as tf
from skimage import io
from skimage.draw import ellipse
import math
import numpy as np

# Rotate the first test image from MNIST:
(xTrain, _), _ = mnist.load_data()
image = xTrain[0]

io.imshow(image)
io.show()

trans1 = tf.AffineTransform(translation=(-14, -14))
rot = tf.AffineTransform(shear=0.5)
trans2 = tf.AffineTransform(translation=(14, 14))
modified = tf.warp(image, (trans1 + (rot + trans2)).inverse)
io.imshow(modified)
io.show()

# Draw a circle
im = np.zeros((200, 200), dtype=np.float32)
elx, ely = ellipse(100, 100, 50, 50)
im[elx, ely] = 1
io.imshow(im)
io.show()
