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
            df (pd.DataFrame): reference DataFrame used to run the missing value analysis.
            time_index (str,bool, optional): references column to be used as index in timeseries df.
                If not provided, VMV will try to infer from index if it is a timeseries or not.
                The index should not be referenced in time_index (it will be tested if left blank).
        """
        super().__init__(df=df)
        self._tests = []
        self._time_index = time_index  # None implies the index
        self._is_time_series = False  # Controls whether the df is treated as time series or not
        self.check_time_index()  # Validates status regarding being a time series or not

    def check_time_index(self):
        """Tries to infer from current time_index column if the dataframe is a timeseries or not.
        If no time_index column was provided tries the index column.
        Overwrites _time_index attribute if the time_index test fails and raises a warning.
        Sets _is_time_series to True if test passes"""
        raises = False
        if self._time_index:
            time_index = self.df[self._time_index]
            raises = True
        else:
            time_index = self.index  # Only tests index if no time_index column is passed
        if isinstance(time_index, pd.DatetimeIndex):
            self._is_time_series = True


        
        
