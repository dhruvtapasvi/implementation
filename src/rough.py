"""This file is for quickly playing with sections of the code. PLEASE GET RID OF IT!"""

from config import routes
from config.DeepDenseAutoencoderConfig import DeepDenseAutoencoderConfig


modelConfig = DeepDenseAutoencoderConfig(routes.CONFIG_ROUTE + "/model/deepDense/deepDense_28x28_ENC_512_512_1024_1024_DEC_512_512_512_512_LAT_512_bce.json")
model = modelConfig.fromConfig()
model.buildModels()
model.summary()
