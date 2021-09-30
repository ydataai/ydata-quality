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


class StringEnum(Enum):

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            up_value = value.upper()

            if up_value in cls.__members__:
                return cls(up_value)

        raise ValueError("%r is not a valid %s" % (value, cls.__name__))
