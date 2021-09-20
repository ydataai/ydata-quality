"""
Implementation of abstract class for Data Quality engines.
"""
from abc import ABC
from collections import Counter
from typing import Optional

import pandas as pd
from numpy import random

from ydata_quality.core.warnings import Priority, QualityWarning
from ydata_quality.utils.auxiliary import infer_df_type, infer_dtypes
from ydata_quality.utils.enum import DataFrameType


class QualityEngine(ABC):
    "Main class for running and storing data quality analysis."

    def __init__(self, df: pd.DataFrame, random_state: Optional[int] = None, label: str = None, dtypes: dict = None):
        self._df = df
        self._df_type = None
        self._warnings = list()
        self._tests = []
        self._label = label
        self._dtypes = dtypes
        self._random_state = random_state

    @property
    def df(self):
        "Target of data quality checks."
        return self._df

    @property
    def label(self):
        "Property that returns the label under inspection."
        return self._label

    @label.setter
    def label(self, label: str):
        if not isinstance(label, str):
            raise ValueError("Property 'label' should be a string.")
        assert label in self.df.columns, "Given label should exist as a DataFrame column."
        self._label = label

    @property
    def dtypes(self):
        "Infered dtypes for the dataset."
        if self._dtypes is None:
            self._dtypes = infer_dtypes(self.df)
        return self._dtypes

    @dtypes.setter
    def dtypes(self, dtypes: dict):
        if not isinstance(dtypes, dict):
            raise ValueError("Property 'dtypes' should be a dictionary.")
        assert all(col in self.df.columns for col in dtypes), "All dtypes keys must be columns in the dataset."
        supported_dtypes = ['numerical', 'categorical']
        assert all(dtype in supported_dtypes for dtype in dtypes.values()), "Assigned dtypes must be in the supported \
broad dtype list: {}.".format(supported_dtypes)
        df_col_set = set(self.df.columns)
        dtypes_col_set = set(dtypes.keys())
        missing_cols = df_col_set.difference(dtypes_col_set)
        if missing_cols:
            _dtypes = infer_dtypes(self.df, skip=df_col_set.difference(missing_cols))
            for col, dtype in _dtypes.items():
                dtypes[col] = dtype
        self._dtypes = dtypes

    @property
    def df_type(self):
        "Infered type for the dataset."
        if self._df_type is None:
            self._df_type = infer_df_type(self.df)
        return self._df_type

    @property
    def random_state(self):
        "Last set random state."
        return self._random_state

    @random_state.setter
    def random_state(self, new_state):
        "Sets new state to random state."
        try:
            self._random_state = new_state
            random.seed(self.random_state)
        except:
            print('An invalid random state was passed. Acceptable values are integers >= 0 or None. Setting to None.')
            self._random_state = None

    def __clean_warnings(self):
        """Deduplicates and sorts the list of warnings."""
        self._warnings = sorted(list(set(self._warnings))) # Sort unique warnings by priority

    def store_warning(self, warning: QualityWarning):
        "Adds a new warning to the internal 'warnings' storage."
        self._warnings.append(warning)

    def get_warnings(self,
                    category: Optional[str] = None,
                    test: Optional[str] = None,
                    priority: Optional[Priority] = None):
        "Retrieves warnings filtered by their properties."
        self.__clean_warnings()
        filtered = [w for w in self._warnings if w.category == category] if category else self._warnings
        filtered = [w for w in filtered if w.test == test] if test else filtered
        filtered = [w for w in filtered if w.priority == Priority(priority)] if priority else filtered
        return filtered  # sort by priority

    @property
    def tests(self):
        "List of individual tests available for the data quality checks."
        return self._tests

    def report(self):
        "Prints a report containing all the warnings detected during the data quality analysis."
        self.__clean_warnings()
        if not self._warnings:
            print('No warnings found.')
        else:
            prio_counts = Counter([warn.priority.value for warn in self._warnings])
            print('Warnings count by priority:')
            print(*(f"\tPriority {prio}: {count} warning(s)" for prio, count in prio_counts.items()), sep='\n')
            print(f'\tTOTAL: {len(self._warnings)} warning(s)')
            print('List of warnings sorted by priority:')
            print(*(f"\t{warn}" for warn in self._warnings), sep='\n')

    def evaluate(self):
        "Runs all the indidividual tests available within the same suite. Returns a dict of (name: results)."
        self._warnings = list() # reset the warnings
        results = {}
        for test in self.tests:
            try: # if anything fails
                results[test] = getattr(self, test)()
            except Exception as exc: # print a Warning and log the message
                print(f'WARNING: Skipping test {test} due to failure during computation.')
                results[test] = "[ERROR] Test failed to compute. Original exception: "+f"{exc}"
        return results
