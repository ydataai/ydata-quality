"""
Implementation of BiasFairness engine to run bias and fairness analysis.
"""

from typing import List, Optional

import pandas as pd
from dython.nominal import compute_associations
from ydata_quality.core import QualityEngine, QualityWarning
from ydata_quality.utils.correlations import filter_associations
from ydata_quality.utils.modelling import (baseline_performance,
                                           performance_per_feature_values)


class BiasFairness(QualityEngine):
    """ Engine to run bias and fairness analysis.

    Tests:
        - Proxy Identification
        - Sensitive Predictability
        - Performance Discrimination
    """

    def __init__(self, df: pd.DataFrame, sensitive_features: List[str], label: Optional[str] = None):
        """
        Args
            df (pd.DataFrame): reference DataFrame used to run the analysis
            sensitive_features (List[str]): features deemed as sensitive attributes
            label (str, optional): target feature to be predicted
        """
        super().__init__(df=df, label=label)
        self._sensitive_features = sensitive_features
        self._tests = ["performance_discrimination", "proxy_identification"]

    @property
    def sensitive_features(self):
        "Returns a list of sensitive features."
        return self._sensitive_features

    def proxy_identification(self, th=0.8):
        """Tests for non-protected features high correlation with sensitive attributes.

        Non-sensitive features can serve as proxy for protected attributes, exposing the data to a possible
        subsequent bias in the data pipeline. High association values indicate that alternative features can
        be used in place of the original sensitive attributes.
        """
        # TODO: multiple thresholds per association type (num/num, num/cat, cat/cat)

        # Compute association measures for sensitive features
        corrs = compute_associations(self.df, num_num_assoc='pearson',nom_nom_assoc='cramer')
        corrs = filter_associations(corrs, th=th, name='association', subset=self.sensitive_features)

        if len(corrs) > 0:
            self.store_warning(
                QualityWarning(
                    test='Proxy Identification', category='Bias&Fairness', priority=2, data=corrs,
                    description=f"Found {len(corrs)} feature pairs of correlation "\
                                 f"to sensitive attributes with values higher than defined threshold ({th})."
            ))
        return corrs


    def sensitive_predictability(self, th=0.8):
        """Trains a baseline classifier to predict sensitive attributes based on remaining features.

        Good performances indicate that alternative features may be working as proxies for sensitive attributes.
        """
        drop_features = self.sensitive_features + [self.label] # features to remove in prediction

        performances = pd.Series(index=self.sensitive_features)
        for feat in performances.index:
            data = self.df.drop(columns=[x for x in drop_features if x != feat]) # drop all except target
            performances[feat] = baseline_performance(df=data, target=feat)
        return performances

    def performance_discrimination(self):
        """Checks for performance disparities for sensitive attributes.

        Get the performance of a baseline model for each feature value of a sensitive attribute.
        High disparities in the performance metrics indicate that the model may not be fair across sensitive attributes.
        """
        if self.label is None:
            print('Argument "label" must be defined to calculate performance discrimination metric. Skipping test.')
            pass

        res = {}
        for feat in self.sensitive_features:
            res[feat] = pd.Series(performance_per_feature_values(df=self.df, feature=feat, target=self.label))
        return res
