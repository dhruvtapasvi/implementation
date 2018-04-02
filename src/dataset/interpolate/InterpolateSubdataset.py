import numpy as np


class InterpolateSubdataset:
    def __init__(self, interpolatedFactorName, left, right, centre=None, outside=None):
        self.__interpolatedFactorName = interpolatedFactorName
        self.__xLeft, self.__yLeft = left
        self.__xRight, self.__yRight = right
        self.__centreSpecified = False
        if centre is not None:
            self.__centreSpecified = True
            self.__xCentre, self.__yCentre = centre
        self.__outsideSpecified = False
        if outside is not None:
            self.__outsideSpecified = True
            self.__xOutside, self.__yOutside = outside

    @property
    def interpolatedFactorName(self):
        return self.__interpolatedFactorName

    @property
    def xLeft(self):
        return self.__xLeft

    @property
    def yLeft(self):
        return self.__yLeft

    @property
    def xRight(self):
        return self.__xRight

    @property
    def yRight(self):
        return self.__yRight

    @property
    def xCentre(self):
        return self.__xCentre

    @property
    def yCentre(self):
        return self.__yCentre

    @property
    def xOutside(self):
        return self.__xOutside

    @property
    def yOutside(self):
        return self.__yOutside

    def centreIsSpecified(self):
        return self.__centreSpecified

    def outsideIsSpecified(self):
        return self.__outsideSpecified
