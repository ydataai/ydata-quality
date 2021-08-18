"""
Auxiliary utility methods, IO, processing, etc.
"""

import json

def test_load_json_path(json_path: str):
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