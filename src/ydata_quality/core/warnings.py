"""
Definition of a data quality warning.
"""

from typing import Any

from pydantic import BaseModel
from ydata_quality.utils.enum import OrderedEnum


class Priority(OrderedEnum):
    """Priorities translate the expected impact of data quality issues.

    Priorities:
      P0: blocks using the dataset
      P1: heavy impact expected
      P2: allows usage but may block human-intelligible insights
      P3: minor impact, aesthetic
    """
    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3

    def __str__(self):
        "Priority {value}: {long description}"
        _descriptions = {
            0: 'blocks using the dataset',
            1: 'heavy impact expected',
            2: 'usage allowed, limited human intelligibility',
            3: 'minor impact, aesthetic'
            }
        return f"Priority {self.value}: {_descriptions[self.value]}"


class QualityWarning(BaseModel):
    """ Details for issues detected during data quality analysis.

    category: name of the test suite (e.g. 'Duplicates')
    test: name of the individual test (e.g. 'Exact Duplicates')
    description: long-text description of the results
    priority: expected impact of data quality warning
    data: sample data
    """

    category: str
    test: str
    description: str
    priority: Priority
    data: Any = None

    #########################
    # String Representation #
    #########################
    def __str__(self):
        return f"[{self.test.upper()}] {self.description} ({str(self.priority)})"

    ########################
    # Comparison Operators #
    ########################
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.priority >= other.priority
        return NotImplemented
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.priority > other.priority
        return NotImplemented
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.priority <= other.priority
        return NotImplemented
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.priority < other.priority
        return NotImplemented

    ##########################
    #  Hashable Definition   #
    ##########################

    def __hash__(self):
        # Hashable definition is needed for storing the elements in a set.
        return hash((self.category, self.test, self.description, self.priority))

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return all(
                (
                self.category == other.category,
                self.test == other.test,
                self.description == other.description,
                self.priority == other.priority
                )
            )
        return NotImplemented
