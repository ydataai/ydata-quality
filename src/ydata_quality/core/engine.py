"""
Implementation of abstract class for Data Quality engines.
"""
from abc import ABC
from typing import Optional

import pandas as pd
from ydata_quality.core.warnings import QualityWarning, Priority
from ydata_quality.utils.modelling import infer_dtypes

class QualityEngine(ABC):
    "Main class for running and storing data quality analysis."

    def __init__(self, df: pd.DataFrame, label: str = None, dtypes: dict = None):
        self._df = df
        self._warnings = set()
        self._tests = []
        self._label = label
        self._dtypes = dtypes

    @property
    def df(self):
        "Target of data quality checks."
        return self._df

    @property
    def warnings(self):
        "Storage of all detected data quality warnings."
        return self._warnings


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
        assert all(col in self.df.columns for col in dtypes), "All dtypes keys \
            must be columns in the dataset."
        supported_dtypes = ['numerical', 'categorical']
        assert all(dtype in supported_dtypes for dtype in dtypes.values()), "Assigned dtypes\
             must be in the supported broad dtype list: {}.".format(supported_dtypes)
        df_col_set = set(self.df.columns)
        dtypes_col_set = set(dtypes.keys())
        missing_cols = df_col_set.difference(dtypes_col_set)
        if missing_cols:
            _dtypes = infer_dtypes(self.df, skip=df_col_set.difference(missing_cols))
            for col, dtype in _dtypes.items():
                dtypes[col] = dtype
        self._dtypes = dtypes

    def store_warning(self, warning: QualityWarning):
        "Adds a new warning to the internal 'warnings' storage."
        self._warnings.add(warning)

    def get_warnings(self,
                    category: Optional[str] = None,
                    test: Optional[str] = None,
                    priority: Optional[Priority] = None):
        "Retrieves warnings filtered by their properties."
        filtered = list(self.warnings) # convert original set
        filtered = [w for w in filtered if w.category == category] if category else filtered
        filtered = [w for w in filtered if w.test == test] if test else filtered
        filtered = [w for w in filtered if w.priority == Priority(priority)] if priority else filtered
        filtered.sort() # sort by priority
        return filtered

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
        results = {}
        for test in self.tests:
            try: # if anything fails
                results[test] = getattr(self, test)()
            except Exception as exc: # print a Warning and log the message
                print(f'WARNING: Skipping test {test} due to failure during computation.')
                results[test] = "[ERROR] Test failed to compute. Original exception: "+f"{exc}"
        return results
