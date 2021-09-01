"""
Implementation of main class for Data Quality checks.
"""
from collections import Counter
from typing import Callable, List, Optional, Union

import pandas as pd

from ydata_quality.core.warnings import Priority, QualityWarning
from ydata_quality.drift import DriftAnalyser
from ydata_quality.duplicates import DuplicateChecker
from ydata_quality.labelling import LabelInspector
from ydata_quality.missings import MissingsProfiler
from ydata_quality.valued_missing_values import VMVIdentifier


class DataQuality:
    "DataQuality contains the multiple data quality engines."

    def __init__(self,
                    df: pd.DataFrame,
                    label: str = None,
                    entities: List[Union[str, List[str]]] = [],
                    vmv_extensions: Optional[list]=[],
                    sample: Optional[pd.DataFrame] = None,
                    model: Callable = None
                    ):
        """
        Engines:
        - Duplicates
        - Missing Values
        - Labelling
        - Valued Missing Values
        - Drift Analysis

        Args:
            df (pd.DataFrame): reference DataFrame used to run the DataQuality analysis.
            label (str, optional): [MISSINGS, LABELLING, DRIFT ANALYSIS] target feature to be predicted.
                                    If not specified, LABELLING is skipped.
            entities: [DUPLICATES] entities relevant for duplicate analysis.
            vmv_extensions: [VALUED MISSING VALUES] A list of user provided valued missing values to append to defaults.
            sample: [DRIFT ANALYSIS] data against which drift is tested.
            model: [DRIFT ANALYSIS] model wrapped by ModelWrapper used to test concept drift.
        """
        self.df = df
        self._warnings = list()
        self._engines = { # Default list of engines
            'duplicates': DuplicateChecker(df=df, entities=entities),
            'missings': MissingsProfiler(df=df, target=label),
            'valued-missing-values': VMVIdentifier(df=df, vmv_extensions=vmv_extensions),
            'drift-analysis': DriftAnalyser(ref=df, sample=sample, label=label, model=model)
        }

        # Engines based on mandatory arguments
        if label is not None:
            self._engines['labelling'] = LabelInspector(df=df, label=label)
        else:
            print('Label is not defined. Skipping LABELLING engine.')

    def __clean_warnings(self):
        """Deduplicates and sorts the list of warnings."""
        self._warnings = sorted(list(set(self._warnings))) # Sort unique warnings by priority

    def get_warnings(self,
                    category: Optional[str] = None,
                    test: Optional[str] = None,
                    priority: Optional[Priority] = None) -> List[QualityWarning]:
        "Retrieves warnings filtered by their properties."
        self.__store_warnings()
        self.__clean_warnings()
        filtered = [w for w in self._warnings if w.category == category] if category else self._warnings
        filtered = [w for w in filtered if w.test == test] if test else filtered
        filtered = [w for w in filtered if w.priority == Priority(priority)] if priority else filtered
        return filtered

    @property
    def engines(self):
        "Dictionary of instantiated engines to run data quality analysis."
        return self._engines

    def __store_warnings(self):
        "Appends all warnings from individiual engines into warnings of DataQuality main class."
        for engine in self.engines.values():
            self._warnings += engine.get_warnings()

    def evaluate(self):
        "Runs all the individual data quality checks and aggregates the results."
        results = {name: engine.evaluate() for name, engine in self.engines.items()}
        return results

    def report(self):
        "Prints a report containing all the warnings detected during the data quality analysis."
        self.__store_warnings() # fetch all warnings from the engines
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
