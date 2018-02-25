import numpy as np


class FilterDatasetLabelPredicate:
    def filter(self, X: np.ndarray, Y: np.ndarray, predicate) -> (np.ndarray, np.ndarray):
        satisfiesCondition = np.array([predicate(y) for y in Y])
        return X[satisfiesCondition], Y[satisfiesCondition]

    def split(self, X: np.ndarray, Y: np.ndarray, predicate) -> ((np.ndarray, np.ndarray), (np.ndarray, np.ndarray)):
        satisfiesCondition = np.array([predicate(y) for y in Y])
        doesNotSatisfyCondition = np.invert(satisfiesCondition)
        return (X[satisfiesCondition], Y[satisfiesCondition]), (X[doesNotSatisfyCondition], Y[doesNotSatisfyCondition])
