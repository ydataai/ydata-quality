"""
Implementation of main class for Data Quality checks.
"""

import pandas as pd

class DataQuality:
    "DataQuality gathers the multiple data quality engines."

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def evaluate(self):
        "Runs all the individual data quality checks and aggregates the results."
        raise NotImplementedError


    def report(self):
        "Returns a full list of warnings retrieved during the Data Quality checks."
        raise NotImplementedError
