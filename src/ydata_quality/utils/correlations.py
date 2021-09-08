"""
Utilities for feature correlations.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def filter_associations(corrs: pd.DataFrame, th: float,
                        name: str = 'corr', subset: Optional[List[str]] = None) -> pd.Series:
    """Filters an association matrix for combinations above a threshold.

    Args:
        corrs (pd.DataFrame): original asssociation matrix (e.g. pandas' corr, dython's compute_associations),
                            shape of (n_feats, n_feats) with association metric (e.g. pearson's correlation, theil's u)
                            as values
        th (float): filter for associations with absolute value higher than threshold
        name (str): name of the association metric
        subset (List[str], optional): list of feature names to subset original association values

    Returns
        corrs (pd.Series): map of feature_pair to association metric value, filtered
    """
    # TODO: replace in high_missing_correlations method of missings engine
    corrs = corrs.copy() # keep original
    np.fill_diagonal(corrs.values, np.nan) # remove the same column pairs
    corrs = corrs[subset] if subset is not None else corrs # subset features
    corrs = corrs[(corrs>th) | (corrs<-th)].melt(ignore_index=False).reset_index().dropna() # subset by threshold
    corrs['features'] = ['_'.join(sorted((i.index, i.variable))) for i in corrs.itertuples()] # create the sorted pairs of feature names
    corrs.drop_duplicates('features', inplace=True) # deduplicate combination pairs
    corrs.sort_values(by='value', ascending=False, inplace=True) # sort by correlation
    corrs = corrs.set_index('features').rename(columns={'value': name})[name] # rename and subset columns
    return corrs
