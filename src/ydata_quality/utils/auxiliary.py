"""
Auxiliary utility methods, IO, processing, etc.
"""

from typing import Union, Tuple

import json

import pandas as pd

def test_load_json_path(json_path: str) -> dict:
    """Tests file existance from given path and attempts to parse as a json dictionary.

    Args:
        json_path (str): A path to a json dictionary.
    Returns:
        json_dict (dict): The json dictionary loaded as Python dictionary.
    """
    if isinstance(json_path, str):
        with open(json_path, 'r') as b_stream:
            data = b_stream.read()
        json_dict = json.loads(data)
    else:
        raise IOError("Expected a path to a json file.")
    return json_dict

def random_split(df: Union[pd.DataFrame, pd.Series], split_size: float, shuffle=True,
                random_state: int=None) -> Tuple[pd.DataFrame]:
    """Shuffles a DataFrame and splits it into 2 partitions according to split_size.
    Returns a tuple with the split first (partition corresponding to split_size, and remaining second).
    Args:
        df (pd.DataFrame): A DataFrame to be split
        split_size (float): Fraction of the sample to be taken
        shuffle (bool): If True shuffles sample rows before splitting
        random_state (int): If an int is passed, the random process is reproducible using the provided seed"""
    assert 0<= split_size <=1, 'split_size must be a fraction, i.e. a float in the [0,1] interval.'
    assert random_state is None or isinstance(random_state, int), 'The random seed must be an integer or None.'
    if shuffle:  # Shuffle dataset rows
        sample = df.sample(frac=1, random_state=random_state)  # An int random_state ensures reproducibility
    split_len = int(sample.shape[0]*split_size)
    split = sample.iloc[:split_len]
    remainder = sample.iloc[split_len:]
    return split, remainder
    