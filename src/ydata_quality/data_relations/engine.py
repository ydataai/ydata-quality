"""
Implementation of DataRelationsDetector engine to run data relations analysis.
"""
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from ydata_quality.core import QualityEngine, QualityWarning
from ydata_quality.utils.correlations import correlation_matrix, partial_correlation_matrix, correlation_plotter, vif_collinearity, chi2_collinearity
from ydata_quality.utils.auxiliary import infer_dtypes, standard_normalize

class DataRelationsDetector(QualityEngine):
    """Main class to run data relations analysis.
    """

    def __init__(self):
        return  # Override the base class init method

    @property
    def tests(self):
        return ["_confounder_detection", "_collider_detection", "_feature_importance", "_inflated_variance_detection"]

    @property
    def dtypes(self):
        return self._dtypes

    @dtypes.setter
    def dtypes(self, df_dtypes: Tuple[pd.DataFrame, dict]):
        df, dtypes = df_dtypes
        if not isinstance(dtypes, dict):
            raise ValueError("Property 'dtypes' should be a dictionary.")
        assert all(col in df.columns for col in dtypes), "All dtypes keys \
            must be columns in the dataset."
        supported_dtypes = ['numerical', 'categorical']
        assert all(dtype in supported_dtypes for dtype in dtypes.values()), "Assigned dtypes\
             must be in the supported broad dtype list: {}.".format(supported_dtypes)
        df_col_set = set(df.columns)
        dtypes_col_set = set(dtypes.keys())
        missing_cols = df_col_set.difference(dtypes_col_set)
        if missing_cols:
            _dtypes = infer_dtypes(df, skip=df_col_set.difference(missing_cols))
            for col, dtype in _dtypes.items():
                dtypes[col] = dtype
        self._dtypes = dtypes

    def evaluate(self, df: pd.DataFrame, dtypes: Optional[dict] = None, label: str=None, corr_th: float=0.8,  vif_th: float=5, p_th: float=0.05, plot: bool=True) -> dict:
        """Runs tests to the validation run results and reports based on found errors.
        Note, we perform standard normalization of numerical features in order to unbias VIF and partial correlation methods.
        This bias correction produces results equivalent to adding a constant feature to the dataset.

        Args:
            df (pd.DataFrame): The Pandas DataFrame on which you want to perform data relations analysis.
            dtypes (Optional[dict]): A dictionary mapping df column names to numerical/categorical dtypes.
                If a full map is not provided it will be determined/completed via inference method.
            label (Optional[str]): A string identifying the label feature column
            corr_th (float): Absolute threshold for high correlation detection. Defaults to 0.8.
            vif_th (float): Variance Inflation Factor threshold for numerical independence test, typically 5-10 is recommended. Defaults to 5.
            p_th (float): Fraction of the right tail of the chi squared CDF defining threshold for categorical independence test. Defaults to 0.05.
            plot (bool): Pass True to produce all available graphical outputs, False to suppress all graphical output.
        """
        assert label in df.columns or not label, "The provided label name does not exist as a column in the dataset"
        self._warnings = [] # reset the warnings to avoid duplicates
        if not dtypes:
            dtypes = {}
        self.dtypes = (df, dtypes)  # Consider refactoring QualityEngine dtypes (df as argument of setter)
        df = standard_normalize(df, dtypes)
        results = {}
        corr_mat, _ = correlation_matrix(df, self.dtypes, True)
        p_corr_mat = partial_correlation_matrix(corr_mat)
        results['Correlations'] = {'Correlation matrix': corr_mat, 'Partial correlation matrix': p_corr_mat}
        if plot:
            correlation_plotter(corr_mat, title='Correlations', symmetric=True)
        if p_corr_mat is not None:
            if plot:
                correlation_plotter(p_corr_mat, title='Partial Correlations', symmetric=True)
            results['Confounders'] = self._confounder_detection(corr_mat, p_corr_mat, corr_th)
            results['Colliders'] = self._collider_detection(corr_mat, p_corr_mat, corr_th)
        else:
            print('[DATA RELATIONS] The partial correlation matrix is not computable for this dataset. Skipping potential confounder and collider detection tests.')
        if label:
            results['Feature Importance'] = self._feature_importance(corr_mat, p_corr_mat, label, corr_th)
        results['High Collinearity'] = self._high_collinearity_detection(df, self.dtypes, label, vif_th, p_th=p_th)
        return results

    def _confounder_detection(self, corr_mat: pd.DataFrame, par_corr_mat: pd.DataFrame, corr_th: float) -> List[Tuple[str, str]]:
        """Detects pairwise variable relationships potentially affected by confounder effects of other covariates.

        Taking the zero order correlations (i.e. without controlling for the influence of any other feature), all
        candidate pairs are compared against the full order partial correlations.
        Zero order coefficient above threshold and partial coefficient below threshold indicate existence of confounding effects."""
        mask = np.ones(corr_mat.shape, dtype='bool')
        mask[np.tril(mask)] = False # Drop pairs below diagonal
        mask[corr_mat.abs()<=corr_th] = False # Drop pairs with zero order correlation below threshold
        mask[par_corr_mat.abs()>corr_th] = False # Drop pairs with correlation after controling all other covariates
        confounded_pairs = [(corr_mat.index[i], corr_mat.columns[j]) for i, j in np.argwhere(mask)]
        if len(confounded_pairs)>0:
            self.store_warning(QualityWarning(
                test='Confounded correlations', category='Data Relations', priority=2, data = confounded_pairs,
                description="Found {} independently correlated variable pairs that disappeared after controling\
 for the remaining variables. This is an indicator of potential confounder effects in the dataset.".format(len(confounded_pairs))))
        return confounded_pairs

    def _collider_detection(self, corr_mat: pd.DataFrame, par_corr_mat: pd.DataFrame, corr_th: float) -> List[Tuple[str, str]]:
        """Detects pairwise variable relationships potentially creating colliding effects with other covariates.

        Taking the zero order correlations (i.e. without controlling for the influence of any other feature), all
        candidate pairs are compared against the full order partial correlations.
        Zero order coefficient below threshold and partial coefficient above threshold indicate existence of collider effects."""
        mask = np.ones(corr_mat.shape, dtype='bool')
        mask[np.tril(mask)] = False # Drop pairs below diagonal
        mask[corr_mat.abs()>corr_th] = False # Drop pairs with zero order correlation above threshold
        mask[par_corr_mat.abs()<=corr_th] = False # Drop pairs with correlation after controling all other covariates
        colliding_pairs = [(corr_mat.index[i], corr_mat.columns[j]) for i, j in np.argwhere(mask)]
        if len(colliding_pairs)>0:
            self.store_warning(QualityWarning(
                test='Collider correlations', category='Data Relations', priority=2, data = colliding_pairs,
                description="Found {} independently uncorrelated variable pairs that showed correlation after\
 controling for the remaining variables. This is an indicator of potential colliding bias with other covariates.".format(len(colliding_pairs))))
        return colliding_pairs

    def _feature_importance(self, corr_mat: pd.DataFrame, par_corr_mat: pd.DataFrame, label: str, corr_th: float) -> pd.DataFrame:
        """Identifies features with high importance.
        Returns all features with absolute correlation to the label higher than corr_th.

        This method returns a summary of all detected important features.
        The summary contains zero, full order partial correlation and a note regarding potential confounding."""
        assert label in corr_mat.columns, "The provided label {} does not exist as a column in the DataFrame.".format(label)
        label_corrs = corr_mat.loc[label].drop(label)
        mask = np.ones(label_corrs.shape, dtype='bool')
        mask[label_corrs.abs()<=corr_th] = False # Drop pairs with zero order correlation below threshold
        important_feats = [label_corrs.index[i][0] for i in np.argwhere(mask)]
        summary = "[FEATURE IMPORTANCE] No important features were found in explaining {}. You might want to try lowering corr_th.".format(label)
        if len(important_feats)>0:
            if par_corr_mat is not None:
                label_pcorrs = par_corr_mat.loc[label].drop(label)
                summary = pd.DataFrame(data={'Correlations': label_corrs.loc[important_feats], 'Partial Correlations': label_pcorrs.loc[important_feats]})
                summary['Note'] = 'OK'
                summary.loc[summary['Partial Correlations'].abs()<corr_th, 'Note'] = 'Potential confounding detected'
            else:
                summary = pd.DataFrame(data={'Correlations': label_corrs.loc[important_feats]})
            summary.sort_values(by='Correlations', ascending=False, inplace=True, key=abs)
        return summary

    def _high_collinearity_detection(self, df: pd.DataFrame, dtypes: dict, label: str=None, vif_th: float= 10., p_th: float=0.05) -> pd.DataFrame:
        """Detects independent variables with high collinearity. Categorical vars and continuous vars are studied as independent sets of variables.
        Variance Inflation Factors are used to study continuous vars collinearity.
        Chi-squared tests are used to test categorical vars collinearity.
        Results are ranked from highest collinearity to lowest and segregated on type of variable.
        """
        vif_scores = vif_collinearity(df, dtypes, p_th, label)
        inflated = vif_scores.loc[vif_scores>vif_th]
        chi2_tests = chi2_collinearity(df, dtypes, p_th,label)
        unique_cats = list(set(list(chi2_tests['Var1'].unique())+list(chi2_tests['Var2'].unique())))
        cat_coll_scores = [(c, chi2_tests[(c == chi2_tests[['Var1','Var2']]).any(axis=1)]['Adjusted Chi2'].mean()) for c in unique_cats]
        cat_coll_scores = [c[0] for c in sorted(cat_coll_scores, key= lambda x: x[1], reverse=True)]
        if len(inflated)>0:
            self.store_warning(QualityWarning(
                test='High Collinearity - Numerical', category='Data Relations', priority=2, data = inflated,
                description="Found {} numerical variables with high Variance Inflation Factor (VIF>{:.1f}).\
 The variables listed in results are highly collinear with other variables in the dataset. These will make model explainability harder and potentially give way to issues like overfitting.\
 Depending on your end goal you might want to remove the highest VIF variables.".format(len(inflated), vif_th)))
        if len(cat_coll_scores)>0:
            # TODO: Determine and sort collinear_cats, as the average aggregated adjusted chi2 tests for each variable in chi2_tests
            # TODO: Merge warning messages (make one warning for the whole test, summarizing findings from the numerical and categorical vars)
            self.store_warning(QualityWarning(
                test='High Collinearity - Categorical', category='Data Relations', priority=2, data = chi2_tests,
                description="Found {} categorical variables with significant collinearity (p-value < {}).\
 The variables listed in results are highly collinear with other variables in the dataset and sorted descending according to propensity. These will make model explainability harder and potentially give way to issues like overfitting.\
 Depending on your end goal you might want to remove variables following the provided order.".format(len(cat_coll_scores), p_th)))
        return {'Numerical': inflated, 'Categorical': cat_coll_scores}
