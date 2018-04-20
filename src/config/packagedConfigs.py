from config.DenseAutoencoderConfig import DenseAutoencoderConfig
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
from config.routes import getConfigRoute


conv_28x28_3_8_256_2_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_28x28_3_8_256_2_bce.json"))
conv_64x64_6_32_2048_64_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_64x64_6_32_2048_64_bce.json"))
conv_64x64_7_8_256_10_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_64x64_7_8_256_10_bce.json"))
conv_64x64_7_16_256_32_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_64x64_7_16_256_32_bce.json"))
conv_64x64_7_32_512_64_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_64x64_7_32_512_64_bce.json"))
conv_96x96_6_8_256_10_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_96x96_6_8_256_10_bce.json"))
conv_96x96_6_16_256_10_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_96x96_6_16_256_10_bce.json"))

dense_28x28_keras_autoencoders_tutorial = DenseAutoencoderConfig(getConfigRoute("model/dense/dense_28x28_keras_autoencoders_tutorial.json"))