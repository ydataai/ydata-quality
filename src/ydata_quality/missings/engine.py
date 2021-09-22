"""
Implementation of MissingProfiler engine to run missing value analysis.
"""
from typing import List, Optional, Union

import numpy as np
from pandas import DataFrame, Series

from ..core import QualityEngine, QualityWarning
from ..utils.modelling import baseline_performance, performance_per_missing_value, predict_missingness


class MissingsProfiler(QualityEngine):
    "Main class to run missing value analysis."

    def __init__(self,
                 df: DataFrame,
                 label: Optional[str] = None,
                 random_state: Optional[int] = None,
                 severity: Optional[str] = None):
        """
        Args:
            df (DataFrame): reference DataFrame used to run the missing value analysis.
            label (str, optional): target feature to be predicted.
            random_state (int, optional): Integer seed for random reproducibility. Default is None.
                Set to None for fully random behavior, no reproducibility.
            severity (str, optional): Sets the logger warning threshold to one of the valid levels [DEBUG, INFO, WARNING, ERROR, CRITICAL]
        """
        super().__init__(df=df, random_state=random_state, label=label, severity=severity)
        self._tests = ["nulls_higher_than", "high_missing_correlations", "predict_missings"]

    def _get_null_cols(self, col: Optional[str] = None) -> List[str]:
        "Returns list of given column or all columns with null values in DataFrame if None."
        return list(self.df.columns[self.null_count(minimal=False) > 0]) if col is None \
            else col if isinstance(col, list) \
            else [col]

    def __get_prediction_type(self):
        "Decide whether to use classification or regression setting, based on label."
        # TODO: Improve prediction type guesstimate based on alternative heuristics (e.g. dtypes, value_counts)

        return 'classification' if len(set(self.df[self.label])) == 2 else 'regression'

    def null_count(self, col: Union[List[str], str, None] = None, normalize=False, minimal=True):
        """Returns the count of null values.

        Args:
            col (optional, str): name of column to calculate nulls. If none, calculates for full dataframe.
            normalize (bool): flag to return nulls as proportion of total rows. Defaults to False.
            minimal (bool): flag to drop zero-nulls when computed for all columns.
        """
        # if col is not provided, calculate for full dataset
        count = self.df.isnull().sum() if col is None else self.df[col].isnull().sum()
        # if normalize, return as percentage of total rows
        count = count / len(self.df) if normalize else count
        # subset
        if col is None and minimal:
            count = count[count > 0]
        return count

    def nulls_higher_than(self, th=0.2):
        "Returns the list of columns with higher missing value percentage than the defined threshold."
        ratios = self.null_count(col=None, normalize=True)
        high_ratios = ratios[ratios >= th]
        if len(high_ratios) > 0:
            self.store_warning(
                QualityWarning(
                    test='High Missings', category='Missings', priority=3, data=high_ratios,
                    description=f"Found {len(high_ratios)} columns with more than {th*100:.1f}% of missing values."
                )
            )
        else:
            high_ratios = None
        return high_ratios

    def missing_correlations(self):
        """Calculate the correlations between missing values in feature values.

        # TODO: Replace standard correlation coefficient by Cramer's V / Theil's U.
        """
        nulls = self.df.loc[:, self.null_count(minimal=False) > 0]  # drop columns without nulls
        return nulls.isnull().corr()

    def high_missing_correlations(self, th: float = 0.5):
        "Returns a list of correlation pairs with high correlation of missing values."

        corrs = self.missing_correlations().abs()        # compute the absolute correlation
        np.fill_diagonal(corrs.values, -1)               # remove the same column pairs
        corrs = corrs[corrs > th].melt(ignore_index=False).reset_index().dropna()  # subset by threshold

        # TODO: For acyclical correlation measures (e.g. Theil's U), store direction as well

        # create the sorted pairs of feature names
        corrs['features'] = ['_'.join(sorted((i.index, i.variable))) for i in corrs.itertuples()]
        corrs.drop_duplicates('features', inplace=True)  # deduplicate combination pairs
        corrs.sort_values(by='value', ascending=False, inplace=True)  # sort by correlation
        corrs = corrs.set_index('features').rename(columns={'value': 'missings_corr'})[
            ['missings_corr']].squeeze()  # rename and subset columns

        if len(corrs) > 0:
            self.store_warning(
                QualityWarning(
                    test='High Missing Correlations', category='Missings', priority=3, data=corrs,
                    description=f"Found {len(corrs)} feature pairs with correlation "
                    f"of missing values higher than defined threshold ({th})."
                )
            )
        return corrs

    def performance_drop(self, col: Union[List[str], str, None] = None, normalize=True):
        """Calculate the drop in performance when the feature values of a given column are missing.

        Performance is measured by "AU-ROC" for binary classification and "Mean Squared Error" for regression.

        Args:
            col (Union[List[str], str, None], optional): reference for comparing performances between valued and missing value instances.
                                    If None, calculates performance_drop for all columns with missing values.
            normalize (bool): performance as ratio over baseline performance achieved for entire dataset.
        """
        # Parse the columns for which to calculate the drop in performance on missings
        cols = self._get_null_cols(col)

        # Guarantee that label is defined. Otherwise skip
        if self.label is None:
            self._logger.warning(
                'Argument "label" must be defined to calculate performance_drop metric. Skipping test.')

        # Guesstimate the prediction type
        prediction_type = self.__get_prediction_type()
        results = DataFrame({
            c: performance_per_missing_value(df=self.df, feature=c, label=self.label, task=prediction_type)
            for c in cols
        })

        # Normalize the results with a baseline performance.
        if normalize:
            baseline = baseline_performance(df=self.df, label=self.label, task=prediction_type)
            results = results / baseline

        return results

    def predict_missings(self, col: Union[List[str], str, None] = None, th=0.8):
        """Calculates the performance score of a baseline model trained to predict missingness of a specific feature.

        Performance is measured on "AU-ROC" for a binary classifier trained to predict occurrence of missing values.
        High performances signal that the occurrence of missing values for a specific feature may be impacted by the
        feature values of all the remaining features.

        Args:
            col (Union[List[str], str, None], optional): reference for predicting occurrence of missing values.
                                    If None, calculates predict_missings for all columns with missing values.
            th (float): performance threshold to generate a QualityWarning.
        """
        # Parse the columns for which to calculate the missingness performance
        cols = self._get_null_cols(col)
        # Calculate the performance for each feature
        results = Series(
                    {c: predict_missingness(df=self.df, feature=c) for c in cols},
                    name='predict_missings', dtype=object
                )

        # Subset for performances above threshold
        high_perfs = results[results > th]

        # Generate a QualityWarning if any high
        if len(high_perfs) > 0:
            self.store_warning(
                QualityWarning(
                    test='Missingness Prediction', category='Missings', priority=2, data=high_perfs,
                    description=f'Found {len(high_perfs)} features with prediction performance \
                        of missingness above threshold ({th}).'
                )
            )
        return results
