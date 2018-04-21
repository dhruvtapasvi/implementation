"""This file is for quickly playing with sections of the code. PLEASE GET RID OF IT!"""

from config import routes
from config.DeepDenseAutoencoderConfig import DeepDenseAutoencoderConfig
from config.ConvolutionAutoencoderConfig import ConvolutionalAutoencoderConfig

from model.loss.binaryCrossEntropyLoss import binaryCrossEntropyLossConstructor
from model.architecture.ConvolutionalDeepIntermediateAutoencoder import ConvolutionalDeepIntermediateAutoencoder
from config.ConvolutionalDeepIntermediateAutoencoderConfig import ConvolutionalDeepIntermediateAutoencoderConfig

# modelConfig = DeepDenseAutoencoderConfig(routes.CONFIG_ROUTE + "/model/deepDense/deepDense_28x28_ENC_512_512_1024_1024_DEC_512_512_512_512_LAT_512_bce.json")
# model = modelConfig.fromConfig()
# model.buildModels()
# model.summary()

# modelConfig = ConvolutionalAutoencoderConfig(routes.CONFIG_ROUTE + "/model/convolutional/conv_64x64_7_16_256_32_bce.json")
# model = modelConfig.fromConfig()
# model.buildModels()
# model.summary()

# model = ConvolutionalDeepIntermediateAutoencoder(binaryCrossEntropyLossConstructor,
#                                                  1.0,
#                                                  (64, 64),
#                                                  6,
#                                                  True,
#                                                  16,
#                                                  [1024, 1024],
#                                                  [1024, 1024],
#                                                  512)
# model.buildModels()
# model.summary()

modelConfig = ConvolutionalDeepIntermediateAutoencoderConfig(routes.CONFIG_ROUTE + "/model/convolutionalDeepIntermediate/conv_64x64_6_16_ENC_1024_DEC_1024_LAT_32_bce.json")
model = modelConfig.fromConfig()
model.buildModels()
model.summary()
