from config.DenseAutoencoderConfig import DenseAutoencoderConfig
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig
from config.DeepDenseAutoencoderConfig import DeepDenseAutoencoderConfig
from config.ConvolutionalDeepIntermediateAutoencoderConfig import ConvolutionalDeepIntermediateAutoencoderConfig
from config.routes import getConfigRoute


conv_28x28_3_8_256_2_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_28x28_3_8_256_2_bce.json"))
conv_64x64_6_32_2048_64_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_64x64_6_32_2048_64_bce.json"))
conv_64x64_7_8_256_10_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_64x64_7_8_256_10_bce.json"))
conv_64x64_7_16_256_32_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_64x64_7_16_256_32_bce.json"))
conv_64x64_7_32_512_64_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_64x64_7_32_512_64_bce.json"))
conv_96x96_6_8_256_10_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_96x96_6_8_256_10_bce.json"))
conv_96x96_6_16_256_10_bce = ConvolutionalAutoencoderConfig(getConfigRoute("model/convolutional/conv_96x96_6_16_256_10_bce.json"))

dense_28x28_keras_autoencoders_tutorial = DenseAutoencoderConfig(getConfigRoute("model/dense/dense_28x28_keras_autoencoders_tutorial.json"))

conv_64x64_6_16_ENC_1024_DEC_1024_LAT_32_bce = ConvolutionalDeepIntermediateAutoencoderConfig(getConfigRoute("model/convolutionalDeepIntermediate/conv_64x64_6_16_ENC_1024_DEC_1024_LAT_32_bce.json"))
conv_28x28_3_8_ENC_1024_DEC_1024_LAT_32_bce = ConvolutionalDeepIntermediateAutoencoderConfig(getConfigRoute("model/convolutionalDeepIntermediate/conv_28x28_3_8_ENC_1024_DEC_1024_LAT_32_bce.json"))

deepDense_28x28_ENC_512x2_1024_DEC_512x3_LAT_32_bce = DeepDenseAutoencoderConfig(getConfigRoute("model/deepDense/deepDense_28x28_ENC_512x2_1024_DEC_512x3_LAT_32_bce.json"))
deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce = DeepDenseAutoencoderConfig(getConfigRoute("model/deepDense/deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce.json"))