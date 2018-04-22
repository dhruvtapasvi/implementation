import numpy as np

from dataset.interpolate.InterpolateSubdataset import InterpolateSubdataset


class CombineInterpolateLoaders:
    def combine(self, interpolateSubdatasets) -> InterpolateSubdataset:
        combinedInterpolateSubdataset = InterpolateSubdataset(
            interpolateSubdatasets[0].interpolatedFactorName,
            (
                np.concatenate([interpolateSubdataset.xLeft for interpolateSubdataset in interpolateSubdatasets]),
                np.concatenate([interpolateSubdataset.yLeft for interpolateSubdataset in interpolateSubdatasets])
            ),
            (
                np.concatenate([interpolateSubdataset.xRight for interpolateSubdataset in interpolateSubdatasets]),
                np.concatenate([interpolateSubdataset.yRight for interpolateSubdataset in interpolateSubdatasets])
            ),
            (
                np.concatenate([interpolateSubdataset.xCentre for interpolateSubdataset in interpolateSubdatasets]),
                np.concatenate([interpolateSubdataset.yCentre for interpolateSubdataset in interpolateSubdatasets])
            ) if interpolateSubdatasets[0].centreIsSpecified() else None,
            (
                np.concatenate([interpolateSubdataset.xOutside for interpolateSubdataset in interpolateSubdatasets]),
                np.concatenate([interpolateSubdataset.yOutside for interpolateSubdataset in interpolateSubdatasets])
            ) if interpolateSubdatasets[0].outsideIsSpecified() else None
        )
        return combinedInterpolateSubdataset
