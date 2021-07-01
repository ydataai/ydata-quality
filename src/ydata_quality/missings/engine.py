"""
Implementation of MissingProfiler engine to run missing value analysis.
"""
from typing import Optional, List
import pandas as pd

from ydata_quality.core import QualityEngine, QualityWarning
from ydata_quality.utils.modelling import predict_missingness, performance_per_missing_value

class MissingsProfiler(QualityEngine):
    "Main class to run missing value analysis."

    def __init__(self, df: pd.DataFrame, target: Optional[str] = None):
        "Run a missing values analysis over a given DataFrame."
        self._df = df
        self._target = target
        self._warnings = set()
        self._tests = ["nulls_higher_than", "high_missing_correlations", "predict_missings"]

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target: str):
        if target not in self.df.columns:
            raise Exception(f'Provided target ({target}) must belong to the dataframe columns ({list(self.df.columns)}).')
        self._target = target

    def _get_null_cols(self, col: Optional[str] = None) -> List[str]:
        "Returns list of given column or all columns with null values in DataFrame if None."
        return list(self.df.columns[self.null_count(minimal=False)>0]) if col is None \
            else col if isinstance(col, list) \
            else [col]

    def __get_prediction_type(self):
        "Decide whether to use classification or regression setting, based on target."
        if len(set(self.df[self.target])) == 2: # binary classification
            return 'classification'
        else:
            return 'regression'

    def null_count(self, col: Optional[str] = None, normalize=False, minimal=True):
        """Returns the count of nulls for a given column. Defaults to full dataframe.

        Args:
            col (optional, str): name of column to calculate nulls. If none, consider all.
            normalize (bool): flag to return nulls as proportion of total rows. Defaults to False.
            minimal (bool): flag to drop zero-nulls when computed for all columns.
        """
        # if col is not provided, calculate for full dataset
        count = self.df.isnull().sum() if col is None else self.df[col].isnull().sum()
        # if normalize, return as percentage of total rows
        count = count / len(self.df) if normalize else count
        # subset
        if col is None and minimal:
            count = count[count>0]
        return count

    def nulls_higher_than(self, th=0.2):
        "Returns the list of columns with higher missing value percentage than the defined threshold."
        ratios = self.null_count(col=None, normalize=True)
        high_ratios = ratios[ratios >= th]
        if len(high_ratios) > 0:
            self._warnings.add(
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
        nulls = self.df.loc[:, self.null_count(minimal=False) > 0] # drop columns without nulls
        return nulls.isnull().corr()

    def high_missing_correlations(self, th: float = 0.5):
        "Returns a list of correlation pairs with high correlation of missing values."

        corrs = self.missing_correlations().abs()        # compute the absolute correlation
        corrs[corrs==1] = -1                             # remove the same column pairs
        corrs = corrs[corrs>th].melt(ignore_index=False).reset_index().dropna() # subset by threshold

        # TODO: For acyclical correlation measures (e.g. Theil's U), store direction as well

        # create the sorted pairs of feature names
        corrs['sorted_pairs'] = ['_'.join(sorted((i.index, i.variable))) for i in corrs.itertuples()]
        corrs.drop_duplicates('sorted_pairs', inplace=True) # deduplicate combination pairs
        corrs.sort_values(by='value', ascending=False, inplace=True) # sort by correlation

        if len(corrs) > 0:
            self._warnings.add(
                QualityWarning(
                    test='High Missing Correlations', category='Missings', priority=3, data=corrs,
                    description=f"Found {len(corrs)} feature pairs with correlation "\
                                 f"of missing values higher than defined threshold ({th})."
                )
            )
        return corrs

    def performance_drop(self, col: Optional[str] = None):
        "Calculate the drop in performance when the feature values of a given column are missing."
        # Parse the columns for which to calculate the drop in performance on missings
        cols = self._get_null_cols(col)
        # Guarantee that target is defined. Otherwise skip
        if self.target is None:
            print('Argument "target" must be defined to calculate performance_drop metric. Skipping test.')
            pass
        prediction_type = self.__get_prediction_type()
        results = {
            c: performance_per_missing_value(df=self.df, feature=c, target=self.target, type=prediction_type)
            for c in cols
        }
        return pd.DataFrame(results)

    def predict_missings(self, col: Optional[str] = None, th=0.8):
        """Calculates the performance score of a baseline model trained to predict missingness of a specific feature.

        If col is not provided, calculate for all the features with missing values.
        """
        # Parse the columns for which to calculate the missingness performance
        cols = self._get_null_cols(col)
        # Calculate the performance for each feature
        results = {c: predict_missingness(df=self.df, feature=c) for c in cols}
        # Subset for performances above threshold
        high_perfs = {k: v for (k,v) in results.items() if v > th}
        if len(high_perfs) > 0:
            self._warnings.add(
                QualityWarning(
                    test='Missingness Prediction', category='Missings', priority=2, data=high_perfs,
                    description=f'Found {len(high_perfs)} features with prediction performance of missingness above threshold ({th}).'
                )
            )
        return results

    def excess_missing_correlations(self):
        """Calculates the difference of feature values correlations between filled and missing values.

        excess_correlation = abs(missings_correlation - correlation)
        """
        raise NotImplementedError
