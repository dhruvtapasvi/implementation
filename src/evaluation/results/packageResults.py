from config.routes import getRecordedResultsRoute
from evaluation.results.FileDictionaryResultsStore import FileDictionaryResultsStore


MODEL_LOSS_RESULTS_FILE = "modelLoss"
modelLossResults = FileDictionaryResultsStore(getRecordedResultsRoute(MODEL_LOSS_RESULTS_FILE))

INTERPOLATION_RESULTS_FILE = "interpolate"
interpolationResults = FileDictionaryResultsStore(getRecordedResultsRoute(INTERPOLATION_RESULTS_FILE))
