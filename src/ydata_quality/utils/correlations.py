"""
Utilities for feature correlations.
"""

from itertools import combinations
from typing import List, Optional
import warnings

import numpy as np
from pandas import DataFrame, Series, crosstab
import scipy.stats as ss
from scipy.stats.distributions import chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import seaborn as sb
import matplotlib.pyplot as plt

from .auxiliary import find_duplicate_columns


def filter_associations(corrs: DataFrame, th: float,
                        name: str = 'corr', subset: Optional[List[str]] = None) -> Series:
    """Filters an association matrix for combinations above a threshold.

    Args:
        corrs (DataFrame): original asssociation matrix (e.g. pandas' corr, dython's compute_associations),
                            shape of (n_feats, n_feats) with association metric (e.g. pearson's correlation, theil's u)
                            as values
        th (float): filter for associations with absolute value higher than threshold
        name (str): name of the association metric
        subset (List[str], optional): list of feature names to subset original association values

    Returns
        corrs (Series): map of feature_pair to association metric value, filtered
    """
    # TODO: replace in high_missing_correlations method of missings engine
    corrs = corrs.copy()  # keep original
    np.fill_diagonal(corrs.values, np.nan)  # remove the same column pairs
    corrs = corrs[subset] if subset is not None else corrs  # subset features
    corrs = corrs[(corrs > th) | (corrs < -th)].melt(ignore_index=False).reset_index().dropna()  # subset by threshold
    corrs['features'] = ['_'.join(sorted((i.index, i.variable)))
                         for i in corrs.itertuples()]  # create the sorted pairs of feature names
    corrs.drop_duplicates('features', inplace=True)  # deduplicate combination pairs
    corrs.sort_values(by='value', ascending=False, inplace=True)  # sort by correlation
    corrs = corrs.set_index('features').rename(columns={'value': name})[name]  # rename and subset columns
    return corrs


def pearson_correlation(col1: np.ndarray, col2: np.ndarray) -> float:
    """Returns Pearson's correlation coefficient for col1 and col2.
    Used for numerical - numerical variable pairs.

    Args:
        col1 (np.ndarray): A numerical column with no null values
        col2 (np.ndarray): A numerical column with no null values"""
    return ss.pearsonr(col1, col2)[0]


def unbiased_cramers_v(col1: np.ndarray, col2: np.ndarray) -> float:
    """Returns the unbiased Cramer's V correlation coefficient for col1 and col2.
    Used for categorical - categorical variable pairs.

    Args:
        col1 (np.ndarray): A categorical column with no null values
        col2 (np.ndarray): A categorical column with no null values"""
    n = col1.size
    contingency_table = crosstab(col1, col2)
    chi_sq = ss.chi2_contingency(contingency_table)[0]
    phi_sq = chi_sq / n
    r, k = contingency_table.shape
    phi_sq_hat = np.max([0, phi_sq - ((r - 1) * (k - 1)) / (n - 1)])
    k_hat = k - np.square(k - 1) / (n - 1)
    r_hat = r - np.square(r - 1) / (n - 1)
    return np.sqrt(phi_sq_hat / np.min([k_hat - 1, r_hat - 1]))  # Note: this is strictly positive


def correlation_ratio(col1: np.ndarray, col2: np.ndarray) -> float:
    """Returns the correlation ratio for col1 and col2.
    Used for categorical - numerical variable pairs.

    Args:
        col1 (np.ndarray): A categorical column with no null values
        col2 (np.ndarray): A numerical column with no null values"""
    uniques = np.unique(col1)
    yx_hat = np.zeros(len(uniques))
    counts = np.zeros(len(uniques))
    for i, value in enumerate(uniques):
        yx = col2[np.where(col1 == value)]
        counts[i] = yx.size
        yx_hat[i] = np.average(yx)
    y_hat = np.average(yx_hat, weights=counts)
    eta_2 = np.sum(np.multiply(counts, np.square(np.subtract(yx_hat, y_hat)))) / np.sum(np.square(np.subtract(col2, y_hat)))  # noqa
    return np.sqrt(eta_2)  # Note: this is strictly positive


def correlation_matrix(df: DataFrame, dtypes: dict, drop_dups: bool = False) -> DataFrame:
    """Returns the correlation matrix.
    The methods used for computing correlations are mapped according to the column dtypes of each pair."""
    corr_funcs = {  # Map supported correlation functions
        ('categorical', 'categorical'): unbiased_cramers_v,
        ('categorical', 'numerical'): correlation_ratio,
        ('numerical', 'numerical'): pearson_correlation,
    }
    # TODO: p-values for every correlation function, to support Data Relations logic
    corr_mat = DataFrame(data=np.identity(n=len(df.columns)), index=df.columns, columns=df.columns)
    p_vals = DataFrame(data=np.ones(shape=corr_mat.shape), index=df.columns, columns=df.columns)
    has_values = df.notnull().values
    df = df.values
    for i, col1 in enumerate(corr_mat):
        dtype1 = dtypes[col1]
        for j, col2 in enumerate(corr_mat):
            if i >= j:
                continue  # Diagonal was filled from the start, lower triangle is equal to top triangle
            dtype2 = dtypes[col2]
            dtype_sorted_ixs = sorted(list(zip([i, j], [dtype1, dtype2])), key=lambda x: x[1])
            key = tuple([col_dtype[1] for col_dtype in dtype_sorted_ixs])
            is_valid = has_values[:, i] & has_values[:, j]  # Valid indexes for computation
            try:
                vals = [df[is_valid, col_dtype[0]] for col_dtype in dtype_sorted_ixs]
                corr = corr_funcs[key](*vals)
            except BaseException:
                corr = None  # Computation failed
            corr_mat.loc[col1, col2] = corr_mat.loc[col2, col1] = corr
    if drop_dups:
        # Find duplicate row lists in absolute correlation matrix
        dup_lists = find_duplicate_columns(corr_mat.abs(), True)
        for col, dup_list in dup_lists.items():
            if col in corr_mat.columns:  # Ensures we will not drop both members of duplicate pairs
                corr_mat.drop(columns=dup_list, index=dup_list, inplace=True)
                p_vals.drop(columns=dup_list, index=dup_list, inplace=True)
    return corr_mat, p_vals


def partial_correlation_matrix(corr_matrix: DataFrame) -> DataFrame:
    """Returns the matrix of full order partial correlations.
    Uses the covariance matrix inversion method."""
    inv_corr_matrix = np.linalg.pinv(corr_matrix)
    diag = np.diag(inv_corr_matrix)
    if np.isnan(diag).any() or (diag <= 0).any():
        return None
    scaled_diag = np.diag(np.sqrt(1 / diag))
    partial_corr_matrix = -1 * (scaled_diag @ inv_corr_matrix @ scaled_diag)
    np.fill_diagonal(partial_corr_matrix, 1)  # Fixing scaling setting the diagonal to -1
    return DataFrame(data=partial_corr_matrix, index=corr_matrix.index, columns=corr_matrix.columns)


def correlation_plotter(mat: DataFrame, title: str = '', symmetric: bool = True):
    """Plots correlation matrix heatmaps.

    Args:
        mat (DataFrame): A correlations matrix (partial or zero order)
        title (str): A string to be used as plot title
        symmetric (bool): True to only plot the lower triangle (symmetric correlation matrix), False to plot all.
        """
    mask = None
    if symmetric:
        mat = mat.iloc[1:, :-1]
        mask = np.zeros_like(mat)
        mask[np.triu_indices_from(mask, 1)] = True

    str_trunc = lambda x: x if len(x) <= 9 else x[:4] + '...' + x[-4:]
    mat.rename(columns=str_trunc, inplace=True)
    plt.figure(figsize=(14, 14))
    ax = sb.heatmap(
        mat, cbar=True, vmin=-1, vmax=1, mask=mask if symmetric else None, annot=True, square=True,
        cmap=sb.diverging_palette(220, 20, as_cmap=True), fmt=".0%")
    if title:
        ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, size=8)
    plt.show()


def vif_collinearity(data: DataFrame, dtypes: dict, label: str = None) -> Series:
    """Computes Variance Inflation Factors for the features of data.
    Disregards the label feature."""
    if label and label in data.columns:
        data = data.drop(columns=label)
    num_columns = [col for col in data.columns if dtypes[col] == 'numerical']
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    vifs = [vif(data[num_columns].values, i) for i in range(len(data[num_columns].columns))]
    warnings.resetwarnings()
    return Series(data=vifs, index=num_columns).sort_values(ascending=False)


def chi2_collinearity(data: DataFrame, dtypes: dict, p_th: float, label: str = None) -> DataFrame:
    """Applies chi-squared test on all combinations of categorical variable pairs in a dataset.
    Disregards the label feature.
    Returns the average of chi-sq statistics found for significant tests (p<p_th) for each categorical variable.
    Returns also the adjusted chi2, i.e. the equivalent chi2 statistic that produces the same p-value in 2 degrees of freedom."""
    cat_vars = sorted([col for col in data.columns if (dtypes[col] == 'categorical' and col != label)])
    combs = list(combinations(cat_vars, 2))
    chis = {'Var1': [],
            'Var2': [],
            'Adjusted Chi2': [],
            'p-value': [],
            'Chi2 stat': [],
            'DoF': []}
    crit_chis = {}
    for comb in combs:
        cont = crosstab(data[comb[0]], data[comb[1]])
        chi, p, dof, _ = ss.chi2_contingency(cont)
        crit_chi = crit_chis.setdefault(dof, chi2.ppf(1 - p_th, dof))
        if chi > crit_chi:
            adj_chi = chi
            if dof != 2:
                adj_chi = chi2.ppf(1 - p, 2)
            for list_, value in zip(chis.values(), [comb[0], comb[1], adj_chi, p, chi, dof]):
                list_.append(value)
    return DataFrame(data=chis).sort_values(by='p-value', ascending=True).reset_index(drop=True)
