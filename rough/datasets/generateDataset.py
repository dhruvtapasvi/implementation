from keras.datasets import mnist
from skimage import transform as tf
from skimage import io
import math

(xTrain, _), _ = mnist.load_data()
image = xTrain[0]

# using skimage
io.imshow(image)
io.show()

trans1 = tf.AffineTransform(translation=(-14, -14))
rot = tf.AffineTransform(shear=0.5)
trans2 = tf.AffineTransform(translation=(14, 14))
modified = tf.warp(image, (trans1 + (rot + trans2)).inverse)
io.imshow(modified)
io.show()
