import pickle
import matplotlib.pyplot as plt

from config.routes import getModelTrainingHistoryRoute, getResultRouteStem, OUT_ROUTE


modelTrainingHistoryRoute = getModelTrainingHistoryRoute("shapesTransformedLimitedRotation_deepDense_64x64_ENC_1024x4_DEC_1024x4_LAT_32_bce")
resultRoute = OUT_ROUTE + "/trainVsValidationLoss.png"


with open(modelTrainingHistoryRoute, "rb") as file:
    modelTrainingHistory = pickle.load(file)
    trainingLossSequence = modelTrainingHistory["loss"]
    validationLossSequence = modelTrainingHistory["val_loss"]

    plt.plot(trainingLossSequence, label="Training Loss")
    plt.plot(validationLossSequence, label="Validation Loss")

    plt.legend()

    plt.xlabel("Number of epochs")

    plt.ylim((0, 2000))
    plt.ylabel("Total Loss Value")

    plt.savefig(resultRoute)
