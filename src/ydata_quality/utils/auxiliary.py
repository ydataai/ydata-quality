"""
Auxiliary utility methods, IO, processing, etc.
"""

from typing import Union, Tuple
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .enum import DataFrameType


def test_load_json_path(json_path: str) -> dict:
    """Tests file existence from given path and attempts to parse as a json dictionary.

    Args:
        json_path (str): A path to a json dictionary.
    Returns:
        json_dict (dict): The json dictionary loaded as Python dictionary.
    """
    if isinstance(json_path, str):
        # pylint: disable=unspecified-encoding
        with open(json_path, 'r') as b_stream:
            data = b_stream.read()
        json_dict = json.loads(data)
    else:
        raise IOError("Expected a path to a json file.")
    return json_dict


def random_split(df: Union[pd.DataFrame, pd.Series], split_size: float,
                 shuffle: bool = True, random_state: int = None) -> Tuple[pd.DataFrame]:
    """Shuffles a DataFrame and splits it into 2 partitions according to split_size.
    Returns a tuple with the split first (partition corresponding to split_size, and remaining second).
    Args:
        df (pd.DataFrame): A DataFrame to be split
        split_size (float): Fraction of the sample to be taken
        shuffle (bool): If True shuffles sample rows before splitting
        random_state (int): If an int is passed, the random process is reproducible using the provided seed"""
    assert random_state is None or (isinstance(random_state, int) and random_state >=
                                    0), 'The random seed must be a non-negative integer or None.'
    assert 0 <= split_size <= 1, 'split_size must be a fraction, i.e. a float in the [0,1] interval.'
    if shuffle:  # Shuffle dataset rows
        sample = df.sample(frac=1, random_state=random_state)
    split_len = int(sample.shape[0] * split_size)
    split = sample.iloc[:split_len]
    remainder = sample.iloc[split_len:]
    return split, remainder


def min_max_normalize(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """Applies min-max normalization to the numerical features of the dataframe.

    Args:
        df (pd.DataFrame): DataFrame to be normalized
        dtypes (dict): Map of column names to variable types"""
    numeric_features = [col for col in df.columns if dtypes.get(col) == 'numerical']
    if numeric_features:
        scaled_data = MinMaxScaler().fit_transform(df[numeric_features].values)
        df[numeric_features] = scaled_data
    return df


def standard_normalize(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """Applies standard normalization (z-score) to the numerical features of the dataframe.

    Args:
        df (pd.DataFrame): DataFrame to be normalized
        dtypes (dict): Map of column names to variable types"""
    numeric_features = [col for col in df.columns if dtypes.get(col) == 'numerical']
    if numeric_features:
        scaled_data = StandardScaler().fit_transform(df[numeric_features].values)
        df[numeric_features] = scaled_data
    return df


def find_duplicate_columns(df: pd.DataFrame, is_close=False) -> dict:
    """Returns a mapping dictionary of columns with fully duplicated feature values.

    Arguments:
        is_close(bool): Pass True to use numpy.isclose instead of pandas.equals."""
    dups = {}
    for idx, col in enumerate(df.columns):  # Iterate through all the columns of dataframe
        ref = df[col]                      # Take the column values as reference.
        for tgt_col in df.columns[idx + 1:]:  # Iterate through all other columns
            if np.isclose(ref, df[tgt_col]).all() if is_close else ref.equals(df[tgt_col]):  # Take target values
                dups.setdefault(col, []).append(tgt_col)  # Store if they match
    return dups


def infer_dtypes(df: Union[pd.DataFrame, pd.Series], skip: Union[list, set] = []):
    """Simple inference method to return a dictionary with list of numeric_features and categorical_features
    Note: The objective is not to substitute the need for passed dtypes but rather to provide expedite inferal between
    numerical or categorical features"""
    infer = pd.api.types.infer_dtype
    dtypes = {}
    as_categorical = ['string',
                      'bytes',
                      'mixed-integer',
                      'mixed-integer-float',
                      'categorical',
                      'boolean',
                      'mixed']
    if isinstance(df, pd.DataFrame):
        for column in df.columns:
            if column in skip:
                continue
            if infer(df[column]) in as_categorical:
                dtypes[column] = 'categorical'
            else:
                dtypes[column] = 'numerical'
    elif isinstance(df, pd.Series):
        dtypes[df.name] = 'categorical' if infer(df) in as_categorical else 'numerical'
    return dtypes


def check_time_index(index: pd.Index) -> bool:
    """Tries to infer from passed index column if the dataframe is a timeseries or not."""
    if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
        return True
    return False


def infer_df_type(df: pd.DataFrame) -> DataFrameType:
    """Simple inference method to dataset type."""
    if check_time_index(df.index):
        return DataFrameType.TIMESERIES
    return DataFrameType.TABULAR
