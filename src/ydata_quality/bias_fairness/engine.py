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
        - Proxy Identification: tests for high correlation between sensitive and non-sensitive features
        - Sensitive Predictability: trains a baseline model to predict sensitive attributes
        - Performance Discrimination: checks for performance disparities on sensitive attributes
    """

    def __init__(self, df: pd.DataFrame, sensitive_features: List[str], label: Optional[str] = None,
        random_state: Optional[int] = None):
        """
        Args
            df (pd.DataFrame): reference DataFrame used to run the analysis
            sensitive_features (List[str]): features deemed as sensitive attributes
            label (str, optional): target feature to be predicted
        """
        super().__init__(df=df, label=label, random_state=random_state)
        self._sensitive_features = sensitive_features
        self._tests = ["performance_discrimination", "proxy_identification",
                        "sensitive_predictability", "sensitive_representativity"]

    @property
    def sensitive_features(self):
        "Returns a list of sensitive features."
        return self._sensitive_features

    def proxy_identification(self, th=0.5):
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


    def sensitive_predictability(self, th=0.5, adjusted_metric=True):
        """Trains a baseline classifier to predict sensitive attributes based on remaining features.

        Good performances indicate that alternative features may be working as proxies for sensitive attributes.
        """
        drop_features = self.sensitive_features + [self.label] # features to remove in prediction

        performances = pd.Series(index=self.sensitive_features)
        for feat in performances.index:
            data = self.df.drop(columns=[x for x in drop_features if x != feat]) # drop all except target
            performances[feat] = baseline_performance(df=data, target=feat, adjusted_metric=adjusted_metric)

        high_perfs = performances[performances>th]
        if len(high_perfs) > 0:
            self.store_warning(
                QualityWarning(
                    test='Sensitive Attribute Predictability', category='Bias&Fairness', priority=3, data=high_perfs,
                    description=f"Found {len(high_perfs)} sensitive attribute(s) with high predictability performance"\
                    f" (greater than {th})."
                )
            )
        return performances

    def performance_discrimination(self):
        """Checks for performance disparities for sensitive attributes.

        Get the performance of a baseline model for each feature value of a sensitive attribute.
        High disparities in the performance metrics indicate that the model may not be fair across sensitive attributes.
        """
        # TODO: support error rate parity metrics (e.g. false positive rate, positive rate)
        if self.label is None:
            print('[BIAS&FAIRNESS] Argument "label" must be defined to calculate performance discrimination metric. Skipping test.')
            pass

        res = {}
        for feat in self.sensitive_features:
            res[feat] = pd.Series(performance_per_feature_values(df=self.df, feature=feat, target=self.label))
        return res


    def sensitive_representativity(self, min_pct: float = 0.01):
        """Checks categorical sensitive attributes minimum representativity of feature values.

        Raises a warning if a feature value of a categorical sensitive attribute is not represented above a min_pct percentage.
        """
        # TODO: Representativity for numerical features
        res = {}
        categorical_sensitives = [k for (k,v) in self.dtypes.items() if (v == 'categorical') & (k in self.sensitive_features)]
        for cat in categorical_sensitives:
            dist = self.df[cat].value_counts(normalize=True) # normalized presence of feature values
            res[cat] = dist # store the distribution
            low_dist = dist[dist<min_pct] # filter for low representativity
            if len(low_dist) > 0:
                self.store_warning(
                QualityWarning(
                    test='Sensitive Attribute Representativity', category='Bias&Fairness', priority=2, data=low_dist,
                    description=f"Found {len(low_dist)} values of '{cat}' sensitive attribute with low representativity"\
                    f" in the dataset (below {min_pct*100:.2f}%)."
                )
            )
        return res
