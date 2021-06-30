"""
Implementation of abstract class for Data Quality engines.
"""
from abc import ABC
import pandas as pd

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

    def evaluate(self):
        "Runs all the indidividual tests available within the same suite. Returns a dict of (name: results)."
        self._warnings = set() # reset the warnings to avoid duplicates
        return {test: getattr(self, test)() for test in self.tests}
