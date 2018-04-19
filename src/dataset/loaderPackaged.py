from dataset.DatasetPackage import DatasetPackage

from dataset.info import MnistInfo as mnistInfo, MnistTransformedInfo as mnistTransformedInfo, NorbInfo as norbInfo, ShapesInfo as shapesInfo

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


baseNorbLoader = NorbLoader(norbInfo.NORB_HOME)
norbPackage = DatasetPackage(
    "norb",
    ScaleBetweenZeroAndOne(baseNorbLoader, *norbInfo.NORB_RANGE),
    ScaleBetweenZeroAndOneInterpolate(NorbInterpolateLoader(baseNorbLoader), *norbInfo.NORB_RANGE)
)


shapesPackage = DatasetPackage(
    "shapes",
    ScaleBetweenZeroAndOne(LoadFromFile(routes.RESOURCE_ROUTE + shapesInfo.HOME, shapesInfo.BASE_IMAGE_SIZE, shapesInfo.BASE_IMAGE_SIZE), *shapesInfo.RANGE),
    ScaleBetweenZeroAndOneInterpolate(ShapesTransformedInterpolateLoader(ShapesBase()), *shapesInfo.RANGE)
)


datasetPackages = [
    mnistPackage,
    mnistTransformedPackage,
    norbPackage,
    shapesPackage
]
