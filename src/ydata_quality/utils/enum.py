"""
Custom implementations of Enums.
"""

from enum import Enum


class PredictionTask(Enum):
    "Enum of supported prediction tasks."
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'


class DataFrameType(Enum):
    "Enum of supported dataset types."
    TABULAR = 'tabular'
    TIMESERIES = 'timeseries'


class OrderedEnum(Enum):
    "Enum with support for ordering."

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
