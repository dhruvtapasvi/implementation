import numpy as np


class Interpolate():
    def interpolateAll(self, left: np.ndarray, right: np.ndarray, intervals):
        """
        Input: N * m * n array
        Output: N * (numIntermediate+1) * m * n array, containing interpolation results
        """
        assert left.shape[0] == right.shape[0]
        length = left.shape[0]
        return np.array([self.interpolateSingle(left[i], right[i], intervals) for i in range(length)])

    def interpolateSingle(self, left, right, intervals):
        return np.array([
            left * ((intervals - step) / intervals) + right * (step / intervals) for step in range(intervals + 1)
        ])
