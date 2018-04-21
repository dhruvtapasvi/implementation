from dataset.DatasetPackage import DatasetPackage

from dataset.info import MnistInfo as mnistInfo, MnistTransformedInfo as mnistTransformedInfo, MnistTransformedInfoLimitedRotation as mnistTransformedInfoLimitedRotation, NorbInfo as norbInfo, ShapesInfo as shapesInfo, ShapesInfoLimitedRotation as shapesInfoLimitedRotation

from dataset.loader.basic.LoadFromFile import LoadFromFile
from dataset.loader.basic.MnistLoader import MnistLoader
from dataset.loader.basic.NorbLoader import NorbLoader
from dataset.loader.preprocess.ScaleBetweenZeroAndOne import ScaleBetweenZeroAndOne
from dataset.loader.basic.ShapesBase import ShapesBase

from dataset.interpolate.basic.MnistInterpolateLoader import MnistInterpolateLoader
from dataset.interpolate.basic.MnistTransformedInterpolateLoader import MnistTransformedInterpolateLoader
from dataset.interpolate.basic.NorbInterpolateLoader import NorbInterpolateLoader
from dataset.interpolate.basic.ShapesTransformedInterpolateLoader import ShapesTransformedInterpolateLoader
from dataset.interpolate.process.ScaleBetweenZeroAndOneInterpolate import ScaleBetweenZeroAndOneInterpolate
from dataset.interpolate.basic.MnistTransformedLimitedRotationInterpolateLoader import MnistTransformedLimitedRotationInterpolateLoader
from dataset.interpolate.basic.ShapesTransformedLimitedRotationInterpolateLoader import ShapesTransformedLimitedRotationInterpolateLoader

from config import routes


baseMnistLoader = MnistLoader()
mnistPackage = DatasetPackage(
    "mnist",
    ScaleBetweenZeroAndOne(baseMnistLoader, *mnistInfo.MNIST_RANGE),
    ScaleBetweenZeroAndOneInterpolate(MnistInterpolateLoader(baseMnistLoader), *mnistInfo.MNIST_RANGE)
)


mnistTransformedPackage = DatasetPackage(
    "mnistTransformed",
    ScaleBetweenZeroAndOne(
        LoadFromFile(mnistTransformedInfo.MNIST_TRANSFORMED_10_HOME, mnistTransformedInfo.IMAGE_DIMENSIONS, mnistTransformedInfo.LABEL_DIMENSIONS),
        *mnistTransformedInfo.RANGE
    ),
    ScaleBetweenZeroAndOneInterpolate(MnistTransformedInterpolateLoader(baseMnistLoader), *mnistTransformedInfo.RANGE)
)


mnistTransformedLimitedRotationPackage = DatasetPackage(
    "mnistTransformedLimitedRotation",
    ScaleBetweenZeroAndOne(
        LoadFromFile(mnistTransformedInfoLimitedRotation.HOME_5, mnistTransformedInfoLimitedRotation.IMAGE_DIMENSIONS, mnistTransformedInfoLimitedRotation.LABEL_DIMENSIONS),
        *mnistTransformedInfoLimitedRotation.RANGE
    ),
    ScaleBetweenZeroAndOneInterpolate(MnistTransformedLimitedRotationInterpolateLoader(baseMnistLoader), *mnistTransformedInfo.RANGE)
)


baseNorbLoader = NorbLoader(norbInfo.NORB_HOME)
norbPackage = DatasetPackage(
    "norb",
    ScaleBetweenZeroAndOne(baseNorbLoader, *norbInfo.NORB_RANGE),
    ScaleBetweenZeroAndOneInterpolate(NorbInterpolateLoader(baseNorbLoader), *norbInfo.NORB_RANGE)
)


shapesPackage = DatasetPackage(
    "shapes",
    ScaleBetweenZeroAndOne(ShapesBase(), *shapesInfo.RANGE),
    ScaleBetweenZeroAndOneInterpolate(ShapesTransformedInterpolateLoader(ShapesBase()), *shapesInfo.RANGE) # Doesn't actually go here
)


shapesTransformedPackage = DatasetPackage(
    "shapesTransformed",
    ScaleBetweenZeroAndOne(LoadFromFile(routes.RESOURCE_ROUTE + shapesInfo.HOME, shapesInfo.IMAGE_DIMENSIONS, shapesInfo.LABEL_DIMENSIONS), *shapesInfo.RANGE),
    ScaleBetweenZeroAndOneInterpolate(ShapesTransformedLimitedRotationInterpolateLoader(ShapesBase()), *shapesInfo.RANGE)
)


shapesTransformedLimitedRotationPackage = DatasetPackage(
    "shapesTransformedLimitedRotation",
    ScaleBetweenZeroAndOne(LoadFromFile(routes.RESOURCE_ROUTE + shapesInfoLimitedRotation.HOME, shapesInfoLimitedRotation.IMAGE_DIMENSIONS, shapesInfoLimitedRotation.LABEL_DIMENSIONS), *shapesInfoLimitedRotation.RANGE),
    ScaleBetweenZeroAndOneInterpolate(ShapesTransformedInterpolateLoader(ShapesBase()), *shapesInfoLimitedRotation.RANGE)
)
