"""
Implementation of LabelInspector engine class to run label quality analysis.
"""
from typing import Union, Optional

from pandas import DataFrame, Series

from ..core import QualityEngine, QualityWarning
from ..utils.auxiliary import infer_dtypes
from ..utils.modelling import (GMM_clustering,
                               estimate_centroid,
                               estimate_sd,
                               normality_test,
                               performance_one_vs_rest,
                               standard_transform)


# pylint: disable=invalid-name
def LabelInspector(df, label, random_state: Optional[int] = None, severity: Optional[str] = None):
    """Runs a label type inference to instantiate the correct label inspector.
    Instantiate this label inspector method to create a Label Inspector.

    Arguments:
            df (DataFrame): reference DataFrame used to run the label analysis.
            label (str, optional): target feature to be predicted.
            random_state (int, optional): Integer seed for random reproducibility. Default is None.
                Set to None for fully random behavior, no reproducibility.
            severity (str, optional): Sets the logger warning threshold to one of the valid levels
                [DEBUG, INFO, WARNING, ERROR, CRITICAL]
        """
    label_dtype = infer_dtypes(df[label])[label]  # Label column dtype inferral

    if label_dtype == 'categorical':
        return CategoricalLabelInspector(df, label, random_state=random_state, severity=severity)

    return NumericalLabelInspector(df, label, random_state=random_state, severity=severity)


class SharedLabelInspector(QualityEngine):
    """Shared structure for Numerical/Categorical Label Inspector"""

    def __init__(self, df: DataFrame, label: str,
                 random_state: Optional[int] = None, severity: Optional[str] = None):
        super().__init__(df=df, label=label, random_state=random_state, severity=severity)
        self._tdf = None

    @property
    def tdf(self):
        "Property that returns the transformed dataset centroids for all (not nan) classes."
        if self._tdf is None:
            self._tdf = self._transform_df()
        return self._tdf

    @staticmethod
    def __get_missing_labels(df: DataFrame, label: str):
        return df[df[label].isna()]

    def _transform_df(self):
        """Selects all observations with a label feature and applies preprocessing transformations.
        Index and column names are preserved.
        Observations with missing label are disregarded."""
        df = self.df[~self.df[self.label].isna()]
        dtypes = self.dtypes
        skip = [self.label] if self.dtypes[self.label] == 'categorical' else []
        tdf, _ = standard_transform(df, dtypes, skip=skip, robust=True)
        return tdf

    def missing_labels(self):
        """Returns observations with missing labels"""
        missing_labels = self.__get_missing_labels(self.df, self.label)
        if len(missing_labels) > 0:
            self.store_warning(
                QualityWarning(
                    test='Missing labels', category='Labels', priority=1, data=missing_labels,
                    description=f"Found {len(missing_labels)} instances with missing labels."
                ))
        else:
            self._logger.info("No missing labels were found.")
            missing_labels = None
        return missing_labels


class CategoricalLabelInspector(SharedLabelInspector):
    """Engine for running analysis on categorical labels.
    Ordinal labels can be handled if passed as categorical."""

    def __init__(self, df: DataFrame, label: str, random_state: Optional[int], severity: Optional[str] = None):
        super().__init__(df=df, label=label, random_state=random_state, severity=severity)
        self._centroids = None
        self._tests = ["missing_labels", "few_labels", "unbalanced_classes",
                       "one_vs_rest_performance", "outlier_detection"]

    @property
    def centroids(self):
        "Property that returns estimated centroids for all (not nan) classes."
        if self._centroids is None:
            self._centroids = self._get_centroids()
        return self._centroids

    def __get_few_labels(self, th=1):
        counts = self._get_label_counts(dropna=True)
        return counts[counts <= th]

    def _get_label_counts(self, dropna=False):
        """Returns a series with unique values of the label column and observation counts
            Args:
                dropna: Controls if NaN (empty) values can be considered as a class of their own"""
        return Series(
            self.df[self.label].value_counts(dropna=dropna).sort_values(ascending=False),
            name='Label counts')

    def few_labels(self, count_th: Union[int, float] = 1):
        """Retrieves labels with a few observations.
        By default returns labels with only one record.
        When a float is passed it is treated as a fraction of the total records."""
        assert count_th > 0 and (isinstance(count_th, int) or 0 < count_th < 1), "\
            count_th must be positive integer or float in the ]0,1[ interval."
        if isinstance(count_th, float):
            count_th = int(count_th * self.df.shape[0])
        few_labels = self.__get_few_labels(count_th)
        if len(few_labels) > 0:
            self.store_warning(
                QualityWarning(
                    test='Few labels', category='Labels', priority=2, data=few_labels,
                    description=f"Found {len(few_labels)} labels with {count_th} or less records."
                ))
        else:
            self._logger.info(f"No labels with {count_th} or less records were found.",)
            few_labels = None
        return few_labels

    def unbalanced_classes(self, slack: float = 0.3):
        """Reports majority/minority classes (above/below a relative count threshold).
        Arguments:
            slack: Margin for alert triggers based on label representativity.
                Slack is linearly adjusted for n>2 classes.
        """
        # TODO: Plot bar chart with observation counts for each class and respective thresholds
        if slack < 0 or slack > 0.5:
            raise ValueError('Slack must be part of the open interval ]0, 0.5[')
        label_counts = self._get_label_counts(dropna=True)  # No label records are not considered
        n_classes = len(label_counts)
        labeled_records = label_counts.sum()  # Total labelled records
        label_ratio = label_counts / labeled_records
        fair_share = 1 / n_classes
        adj_slack = slack * (2 / n_classes)  # Adjust slack depending on number of classes
        label_excess = (label_ratio - fair_share)[abs(label_ratio - fair_share) > adj_slack]
        data = {}
        if len(label_excess) != 0:
            for _class, excess in label_excess.items():
                folder = 'Under-represented'  # By default
                if excess > 0:
                    folder = 'Over-represented'
                data.setdefault(folder, {})[_class] = self.df[self.df[self.label] == _class]
            self.store_warning(
                QualityWarning(
                    test='Unbalanced Classes', category='Labels', priority=2,
                    data=data,
                    description=f"""
                    Classes {set(data['Under-represented'].keys())} \
                        are under-represented each having less than {fair_share-adj_slack:.1%} of total instances. \
                    Classes {set(data['Over-represented'].keys())} \
                        are over-represented each having more than {fair_share+adj_slack:.1%} of total instances
                    """))
        else:
            self._logger.info("No unbalanced classes were found.")
            return None
        return label_excess.index

    def one_vs_rest_performance(self, slack: float = 0):
        """Performs one vs rest classification over each label class.
        Returns a series with Area Under Curve for each label class.
        Slack defines a proportion for the record weighted average of performances as a tolerance.
        Any binary classifier that scores below the average minus tolerance triggers a warning.
        """
        # TODO: Plot ROC curve
        assert 0 <= slack <= 1, "Argument th is expected to be a float in the [0,1] interval"
        _class_counts = self._get_label_counts(dropna=True)
        _class_counts = _class_counts[_class_counts > 1]
        results = {
            _class: performance_one_vs_rest(df=self.tdf, label_feat=self.label,
                                            _class=_class, dtypes=self.dtypes)
            for _class in _class_counts.index
        }
        record_weighted_avg = sum([perf * _class_counts[_class] for _class, perf in results.items()])
        record_weighted_avg = (1 / _class_counts.sum()) * record_weighted_avg
        threshold = (1 - slack) * record_weighted_avg
        poor_performers = {_class: perf for _class, perf in results.items() if perf < threshold}
        if len(poor_performers) > 0:
            self.store_warning(
                QualityWarning(
                    test='One vs Rest Performance', category='Labels', priority=2,
                    data=Series(poor_performers),
                    description="Classes {} performed under the {:.1%} AUROC threshold. \
The threshold was defined as an average of all classifiers with {:.0%} slack.".format(
                        set(poor_performers.keys()), threshold, slack)
                ))
        return Series(results)

    def _get_centroids(self):
        """Produces a centroid estimation for observations grouped by label value.
        Centroids are estimated using the normalized dataset."""
        label_counts = self._get_label_counts(dropna=True)
        centroids = DataFrame(self.tdf.iloc[:len(label_counts)],
                                 columns=self.tdf.columns, index=label_counts.index)
        for i, _class in enumerate(label_counts.index):
            records = self.tdf[self.tdf[self.label] == _class]
            centroids.iloc[i] = estimate_centroid(records, self.dtypes)
        return centroids

    def _get_class_sds(self):
        """Estimates the STDev of intra cluster distances to the centroid over each class.
        Returns:
            sds: A series with the intra cluster distances of each class
            sd_distances: A dictionary with the distances of each point to its centroid (key).
                Distances are scaled by the corresponding stdev of the intra cluster distances"""
        sd_distances = {}
        for _class in self.centroids.index:
            sds = sd_distances.setdefault(_class, {})
            records = self.tdf[self.tdf[self.label] == _class].drop(self.label, axis=1)
            centroid = self.centroids.loc[_class].drop(self.label).values.flatten()
            sds['SD'], sds['Scaled Distances'] = estimate_sd(records, centroid, dtypes=self.dtypes)
        return sd_distances

    def outlier_detection(self, th=3):
        """Provides a dictionary ordered by label values and identifying potential outliers.
        Outliers are defined as points with distance to group centroid bigger than a threshold.
        The threshold is defined in Standard Deviations of the intra-cluster distances."""
        sd_distances = self._get_class_sds()
        potential_outliers = 0
        data = {}
        for _class, sds in sd_distances.items():
            sd_distances = sds['Scaled Distances'][sds['Scaled Distances'] > th]
            new_outliers = len(sd_distances)
            if new_outliers > 0:
                potential_outliers += new_outliers
                data.setdefault(_class, self.df.loc[sd_distances.index])
        if potential_outliers > 0:
            self.store_warning(
                QualityWarning(
                    test='Outlier Detection', category='Labels', priority=2, data=data,
                    description=f"""
                    Found {potential_outliers} potential outliers across {len(data.keys())} classes. \
                    A distance bigger than {th} standard deviations of intra-cluster distances \
                    to the respective centroids was used to define the potential outliers.
                    """
                ))
        return data


class NumericalLabelInspector(SharedLabelInspector):
    "Engine for running analyis on numerical labels."

    def __init__(self, df: DataFrame, label: str, random_state, severity: Optional[str] = None):
        super().__init__(df=df, label=label, random_state=random_state, severity=severity)
        self._tests = ["missing_labels", "test_normality", "outlier_detection"]

    def _GMM_clusters(self, max_clusters):
        """Separates the dataset into a Gaussian Mixture Model cluster optimized nbins.
        Clustering is done only with the label column values."""
        sorted_vals = self.tdf[self.label].sort_values().copy()
        search_space = range(1, max_clusters)
        AICs = [None for k in search_space]
        labels = {k: None for k in search_space}
        for i, k in enumerate(search_space):
            labels[k], AICs[i] = GMM_clustering(sorted_vals.values.reshape(-1, 1), k)
        ideal_k = list(labels.keys())[AICs.index(min(AICs))]
        return Series(labels[ideal_k], index=sorted_vals.index)

    def outlier_detection(self, th: float = 3., use_clusters=False, max_clusters: int = 5):
        """Detects outliers based on standard deviation of the label feature.
        Estimates the median value and standard deviation for the label.
        Signals all values beyond th standard deviations from the median as potential outliers.
        Arguments:
            th: threshold measured in cluster standard deviations
            use_clusters: Set to True in order to detect outliers inside each proposed cluster.
                Set to False to use a unimodal outlier detection strategy (default)
            max_clusters: To take effect must be used with use_clusters passed as True.
                Defines search space upper bound for number of clusters."""
        if use_clusters:
            cluster_labels = self._GMM_clusters(max_clusters)
        else:
            cluster_labels = Series('full_dataset', index=self.tdf.index)
        clusters = cluster_labels.unique()
        potential_outliers = {}
        for cluster in clusters:
            values = self.tdf[self.label][cluster_labels == cluster].copy()
            if len(values) == 1:  # Single element clusters are automatically flagged as potential outliers
                potential_outliers[cluster] = self.df.loc[values.index]
            else:  # Look for deviant elements inside clusters
                median = values.median()
                std = values.std()
                abs_deviations = ((values - median) / std).abs()
                cluster_outliers = self.df.loc[abs_deviations[abs_deviations > th].index]
                if len(cluster_outliers) > 0:
                    potential_outliers[cluster] = cluster_outliers
        if len(potential_outliers) > 0:
            total_outliers = sum([cluster_outliers.shape[0] for cluster_outliers in potential_outliers.values()])
            coverage_string = "{} clusters".format(len(clusters)) if use_clusters else "the full dataset"
            self.store_warning(
                QualityWarning(
                    test='Outlier Detection', category='Labels', priority=2, data=potential_outliers,
                    description=f"""
                    Found {total_outliers} potential outliers across {coverage_string}. \
                    A distance bigger than {th} standard deviations of intra-cluster distances \
                    to the respective centroids was used to define the potential outliers."""
                ))
        return potential_outliers

    def test_normality(self, p_th=5e-3):
        """Runs a normal distribution test on the label column.
        If passes data is normally distributed.
        If it fails, retries with a battery of transformations.
        """
        vals = self.tdf[self.label].copy()
        test_result, transform, pstat = normality_test(vals, p_th=p_th)
        if test_result:
            if transform is None:
                self._logger.info("The label values appears to be normally distributed.")
            else:
                self._logger.info("The %s transform appears to be able to normalize the label values.", transform)
                self.store_warning(
                    QualityWarning(
                        test='Test normality', category='Labels', priority=2, data=vals,
                        description=f"The label distribution as-is failed a normality test. \
Using the {transform} transform provided a positive normality test with a p-value statistic of {pstat:.2f}"
                    ))
        else:
            self._logger.warning("""
            It was not possible to normalize the label values.
            See the data quality warning message for additional context.
            """)
            self.store_warning(
                QualityWarning(
                    test='Test normality',
                    category='Labels',
                    priority=1,
                    data=vals,
                    description="""
                    The label distribution failed to pass a normality test as-is and following a battery of transforms.
It is possible that the data originates from an exotic distribution, there is heavy outlier presence or it is \
multimodal. Addressing this issue might prove critical for regressor performance.
"""
                ))
