"""
Implementation of abstract class for Data Quality engines.
"""
from abc import ABC
from typing import Optional

import pandas as pd
from ydata_quality.core import QualityWarning
from ydata_quality.core.warnings import Priority
from ydata_quality.utils.context import noprint

class QualityEngine(ABC):
    "Main class for running and storing data quality analysis."

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._warnings = set()
        self._tests = []

    @property
    def df(self):
        "Target of data quality checks."
        return self._df

    @property
    def warnings(self):
        "Storage of all detected data quality warnings."
        return self._warnings

    def store_warning(self, warning: QualityWarning):
        "Adds a new warning to the internal 'warnings' storage."
        self._warnings.add(warning)

    def get_warnings(self,
                    category: Optional[str] = None,
                    test: Optional[str] = None,
                    priority: Optional[Priority] = None):
        "Retrieves warnings filtered by their properties."
        filtered = self.warnings # original set
        filtered = [w for w in filtered if w.category == category] if category else filtered
        filtered = [w for w in filtered if w.test == test] if test else filtered
        filtered = [w for w in filtered if w.priority == Priority(priority)] if priority else filtered
        return set(filtered)

    @property
    def tests(self):
        "List of individual tests available for the data quality checks."
        return self._tests

    def report(self):
        "Prints a report containing all the warnings detected during the data quality analysis."
        # TODO: Provide a count of warnings by priority
        self._warnings = set(sorted(self._warnings)) # Sort the warnings by priority
        for warn in self.warnings:
            print(warn)

    @noprint
    def evaluate(self):
        "Runs all the indidividual tests available within the same suite. Returns a dict of (name: results)."
        self._warnings = set() # reset the warnings to avoid duplicates
        return {test: getattr(self, test)() for test in self.tests}
