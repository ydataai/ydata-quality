"""
Implementation of Valued Missing Values Identifier engine class to run valued missing value analysis.
"""

from typing import List, Optional, Union

import pandas as pd

from ydata_quality.core import QualityEngine, QualityWarning


class VMVIdentifier(QualityEngine):
    "Engine for running analyis on valued missing values."

    def __init__(self, df: pd.DataFrame, time_index: Optional[str] = None):
        """
        Args:
            df (pd.DataFrame): DataFrame used to run the missing value analysis.
            time_index (str, optional): references column to be used as index in timeseries df.
                Provide only to override default timeseries inference on the dataset index.
        """
        super().__init__(df=df)
        self._tests = ["odd_dtypes"]
        self._time_index = time_index  # None implies the index
        self._is_time_series = self._check_time_index()  # Validates status regarding being a time series or not
        self._missing_mask = self._get_missing_mask()  # Will help to make some methods more efficient

    def _check_time_index(self):
        """Tries to infer from current time_index column if the dataframe is a timeseries or not.
        Tries index colum by default if time_index argument was not provided.
        Raises warning if a passed time_index test fails, will not raise on default index infer.
        Sets _is_time_series to True if the test passes"""
        raises = False if self._time_index is None else True  # If no _time_index is passed we don't raise warning
        if self._time_index  and self._time_index != '__index':
            time_index = self.df[self._time_index]
        else:
            time_index = self.df.index  # Only tests index if no time_index column is passed
        if isinstance(time_index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
            return True
        # Here we could still try a conversion method to pd.DatetimeIndex but avoiding corner 
        # cases does not seem trivial. Best to have the user provide us with a valid index.
        if raises:
            self._warnings.add(
                QualityWarning(
                    test='Check Time Index', category='Valued Missing Duplicates', priority=2, data=time_index,
                    description=f"The provided column {self._time_index} is not a valid time series index type."
                )
            )
        return False

    def _get_missing_mask(self):
        """Returns a missing mask for the full dataframe."""
        return self.df.isna()

    def flatlines(self):
        """Iterates over columns of the dataset looking for flatlines (adjacent duplicate element sequences)"""
        df = self.df.copy()
        df.rese
        for column in self.df.columns
    
    def odd_dtypes(self):
        """Infers dtype for the columns.
        Based on the infered type will or will not perform element dtype inference for the uniques of the column."""
        infer = pd.api.types.infer_dtype
        to_inspect = {}
        for column in self.df.columns:
            col_dtype = infer(self.df[column])
            if "mixed" in col_dtype:
                to_inspect[column] = col_dtype
            else:
                continue
        for column, dtype in to_inspect:

        
