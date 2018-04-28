import pandas as pd
import pickle


TRAINING_HISTORY_ROOT = "./modelTrainingHistory/"
TRAINING_HISTORY_FILE_NAMES = [
    "mnist_conv_28x28_3_8_ENC_1024_DEC_1024_LAT_32_bce.history.p",
    "mnist_deepDense_28x28_ENC_512x2_1024_DEC_512x3_LAT_32_bce.history.p",
    "norb_deepDense_96x96_ENC_1024_2048_2048_DEC_2048_2048_1024_LAT_32_bce.history.p",
    "shapesTransformedLimitedRotation_deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce.history.p"
]


for trainingHistoryFileName in TRAINING_HISTORY_FILE_NAMES:
    print(trainingHistoryFileName)
    trainingHistoryFilePath = TRAINING_HISTORY_ROOT + trainingHistoryFileName

    with open(trainingHistoryFilePath, "rb") as trainingHistoryFile:
        trainingHistory = pickle.load(trainingHistoryFile)
        df = pd.DataFrame(trainingHistory)
        # display(df.describe(percentiles=[0.25 * i for i in range(4)] + [0.95, 0.99]))
        df.plot(figsize=(8, 6))
        print(df)
