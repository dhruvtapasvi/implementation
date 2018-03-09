from abc import abstractclassmethod, ABCMeta


class ConfigParser(metaclass=ABCMeta):
    @abstractclassmethod
    def fromConfig(self):
        raise NotImplementedError
