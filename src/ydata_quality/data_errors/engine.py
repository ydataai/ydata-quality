"""
Implementation of DataErrorSearcher engine to run data error analysis.
"""
from typing import Optional, Union

import pandas as pd
from ydata_quality.core import QualityEngine, QualityWarning


class DataErrorSearcher(QualityEngine):
    """Main class to run data error analysis.

    Methods:
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the engine properties and lists tests for automated evaluation.
        Args:
            df (pd.DataFrame): DataFrame used to run data error analysis.
        """
        super().__init__(df=df)
