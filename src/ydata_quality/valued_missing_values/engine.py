"""
Implementation of Valued Missing Values Identifier engine class to run valued missing value analysis.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from ydata_quality.core import QualityEngine, QualityWarning


class VMVIdentifier(QualityEngine):
    "Engine for running analyis on valued missing values."

    def __init__(self, df: pd.DataFrame, VMV_extensions: list=[]):
        """
        Args:
            df (pd.DataFrame): DataFrame used to run the missing value analysis.
            time_index (str, optional): references column to be used as index in timeseries df.
                Provide only to override default timeseries inference on the dataset index.
        """
        super().__init__(df=df)
        self._tests = ["flatlines", "predefined_valued_missing_values"]
        self._flatline_index = {}
        self._default_VMVs = set("?", "UNK", "Unknown", "N/A", "NA", "", "(blank)").union(
            set(VMV_extensions))

    @staticmethod
    def __get_flatline_index(column: pd.Series):
        """Returns an index for flatline events on a passed column.
        A flatline event is any full sequence of repeated values on the column.
        The returned index is a compact representation of all occurrences of flatline events.
        Returns a DataFrame with index equal to the index of first element in the event,
        a tail column identifying the last element of the sequence and a length column."""
        column.fillna('__filled')  # NaN values were not being treated as a sequence
        cum_differs_previous = column.ne(column.shift()).cumsum()
        sequence_groups = column.index.to_series().groupby(cum_differs_previous)
        data = {'length': sequence_groups.count().values,
        'ends': sequence_groups.last().values}
        return pd.DataFrame(data, index=sequence_groups.first().values).query('lengths > 1')

    def flatlines(self, th: int=5, skip: list=[]):
        """Checks the flatline index for flat sequences of length over a given threshold.
        Raises warning indicating columns with flatline events and total flatline events in the dataframe.
        Arguments:
            th: Defines the minimum length required for a flatline event to be reported.
            skip: List of columns that will not be target of search for flatlines.
                Pass '__index' in skip to skip looking for flatlines at the index."""
        df = self.df.copy()  # Index will not be covered in column iteration
        df[self.__default_index_name] = df.index  # Index now in columns to be processed next
        flatlines = {}
        for column in df.columns:  # Compile flatline index
            if column in skip:
                continue  # Column not requested
            flt_index = self._flatline_index.setdefault(column,
                self.__get_flatline_index(df[column]))
            flts = flt_index.loc[flt_index['length']>th]
            if len(flts) > 0:
                flatlines[column] = flts
        if len(flatlines)>0:  # Flatlines detected
            total_flatlines = [flts.shape[0] for flts in flatlines.values()]
            self._warnings.add(
                QualityWarning(
                    test='Flatlines', category='Valued Missing Values', priority=2, data=flatlines,
                    description=f"Found {total_flatlines} flatline events with a minimun length of {th} among the columns {set(flatlines.keys())}."
            ))
            return flatlines
        else:
            print("[FLATLINES] No flatline events with a minimum length of {} were found.")

    def predefined_valued_missing_values(self, skip: list=[]):
        """Runs a check against a list of predefined Valued Missing Values.
        Raises warning based on the existance of these values.
        Returns a DataFrame with count distribution for each predefined type over each column.
        Arguments:
            skip: List of columns that will not be target of search for VMVs.
                Pass '__index' in skip to skip looking for flatlines at the index."""
        df = self.df.copy()  # Index will not be covered in column iteration
        df[self.__default_index_name] = df.index  # Index now in columns to be processed
        check_cols = set(df.columns).difference(set(skip))
        df = df[check_cols]
        VMVs = pd.DataFrame(index=self.predefined_valued_missing_values, columns=check_cols)
        for VMV in self.predefined_valued_missing_values:
            VMVs.loc[VMV] = (df==VMV).sum()
        VMVs.drop((VMVs.sum()==0).index, axis=1)
        if VMVs.empty:
            print("[PREDEFINED VALUED MISSING VALUES] No predefined VMVs from  the set {} were found in the dataset.".format(
                self.predefined_valued_missing_values
            ))
        else:
            self._warnings.add(
                QualityWarning(
                    test='Predefined Valued Missing Values', category='Valued Missing Values', priority=2, data=VMVs,
                    description=f"Found {total_flatlines} VMVs in the dataset."
            ))
            return VMVs






        
