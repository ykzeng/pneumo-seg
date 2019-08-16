from abc import abstractmethod
import sys
import abc
import logging
import datetime

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Model(ABC):
    def __init__(self):
        self.endDate = None
        self.model = None
        self.features = None
        self.ccode = None

    @abstractmethod
    def nn_setup(self):
        pass

    @abstractmethod
    def train(self, x, y, modelPath=False):
        pass

    """predict() the specified time periods, must be ran after train()

    Parameters
    ----------
    periods : int
        how many days we are going to predict beyond the historical data
    resPath : string, optional
        the path in filesystem to save prediction results

    Returns
    -------
    pandas.DataFrame
        a DataFrame containing at least the two columns: [dates] and [yhat] (predicted value)
    """
    @abstractmethod
    def predict(self, x):
        pass

    # def persist(self, path):
    #     pass
