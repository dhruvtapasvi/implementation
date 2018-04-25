import pandas as pd
import pickle


TRAINING_HISTORY_ROOT = "./modelTrainingHistory/"
TRAINING_HISTORY_FILE_NAMES = [
    "mnistTransformedLimitedRotation_deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce.history.p"
]


for trainingHistoryFileName in TRAINING_HISTORY_FILE_NAMES:
    trainingHistoryFilePath = TRAINING_HISTORY_ROOT + trainingHistoryFileName

    with open(trainingHistoryFilePath, "rb") as trainingHistoryFile:
        trainingHistory = pickle.load(trainingHistoryFile)
        df = pd.DataFrame(trainingHistory)
        # display(df.describe(percentiles=[0.25 * i for i in range(4)] + [0.95, 0.99]))
        df.plot(figsize=(8, 6))
