"""
Implementation of LabelInspector engine class to run label quality analysis.
"""
from typing import Union

import pandas as pd

from ydata_quality.core import QualityEngine, QualityWarning
from ydata_quality.utils.modelling import (GMM_clustering, elbow,
                                           estimate_centroid, estimate_sd,
                                           infer_dtypes, kmeans,
                                           normality_test,
                                           performance_one_vs_rest,
                                           standard_transform)


def LabelInspector(df, label):
    """Instantiate this label inspector class.
    Runs a label type inference to instantiate the correct label inspector."""
    label_dtype = infer_dtypes(df[label])[label]  # Label column dtype inferral
    if label_dtype == 'categorical':
        return CategoricalLabelInspector(df, label)
    else:
        return NumericalLabelInspector(df, label)

class SharedLabelInspector(QualityEngine):
    """Shared structure for Numerical/Categorical Label Inspector"""

    def __init__(self, df: pd.DataFrame, label: str):
        super().__init__(df)  # Runs init from the Quality Engine
        self._label = label
        self._dtypes = infer_dtypes(self.df)
        self._tdf = None

    @property
    def label(self):
        "Property that returns the label under inspection."
        return self._label

    @label.setter
    def label(self, label: str):
        if not isinstance(label, str):
            raise ValueError("Property 'label' should be a string.")
        assert label in self.df.columns, "Given label should exist as a DataFrame column."
        self._label = label

    @property
    def dtypes(self):
        "Property that returns infered dtypes for the dataset."
        return self._dtypes

    @dtypes.setter
    def dtypes(self, dtypes: dict):
        if not isinstance(dtypes, dict):
            raise ValueError("Property 'dtypes' should be a dictionary.")
        assert all(col in self.df.columns for col in dtypes), "All dtypes keys \
            must be columns in the dataset."
        supported_dtypes = ['numerical', 'categorical']
        assert all(dtype in supported_dtypes for dtype in dtypes.values()), "Assigned dtypes\
             must be in the supported broad dtype list: {}.".format(supported_dtypes)
        df_col_set = set(self.df.columns)
        dtypes_col_set = set(dtypes.keys())
        missing_cols = df_col_set.difference(dtypes_col_set)
        if missing_cols:
            _dtypes = infer_dtypes(self.df, skip=df_col_set.difference(missing_cols))
            for col, dtype in _dtypes.items():
                dtypes[col] = dtype
        self._dtypes = dtypes

    @property
    def tdf(self):
        "Property that returns the transformed dataset centroids for all (not nan) classes."
        if self._tdf is None:
            self._tdf = self._transform_df()
        return self._tdf

    @staticmethod
    def __get_missing_labels(df: pd.DataFrame, label: str):
        return df[df[label].isna()]

    def _get_data_types(self):
        """Makes a guesstimate for the column types.
        Used to control the distance metrics and preprocessing pipelines"""
        dtypes = infer_dtypes(self.df)
        return dtypes

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
            self._warnings.add(
                QualityWarning(
                    test='Missing labels', category='Labels', priority=1, data=missing_labels,
                    description=f"Found {len(missing_labels)} instances with missing labels."
            ))
        else:
            print("[MISSING LABELS] No missing labels were found.")
            missing_labels = None
        return missing_labels


class CategoricalLabelInspector(SharedLabelInspector):
    """Engine for running analysis on categorical labels.
    Ordinal labels can be handled if passed as categorical."""

    def __init__(self, df: pd.DataFrame, label: str):
        super().__init__(df, label)
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
        return counts[counts<=th]

    def _get_label_counts(self, dropna=False):
        """Returns a series with unique values of the label column and observation counts
            Args:
                dropna: Controls if NaN (empty) values can be considered as a class of their own"""
        return pd.Series(
            self.df[self.label].value_counts(dropna=dropna).sort_values(ascending=False),
            name='Label counts')

    def few_labels(self, count_th: Union[int, float] = 1):
        """Retrieves labels with a few observations.
        By default returns labels with only one record.
        When a float is passed it is treated as a fraction of the total records."""
        assert count_th>0 and (isinstance(count_th, int) or 0<count_th<1), "\
            count_th must be positive integer or float in the ]0,1[ interval."
        if isinstance(count_th, float):
            count_th = int(count_th*self.df.shape[0])
        few_labels = self.__get_few_labels(count_th)
        if len(few_labels) > 0:
            self._warnings.add(
                QualityWarning(
                    test='Few labels', category='Labels', priority=2, data=few_labels,
                    description=
                    "Found {} labels with {} or less records.".format(len(few_labels), count_th)
            ))
        else:
            print("[FEW LABELS] No labels with {} or less records were found.".format(count_th))
            few_labels = None
        return few_labels

    def unbalanced_classes(self, slack: float = 0.3):
        """Reports majority/minority classes (above/below a relative count threshold).
        Arguments:
            slack: Margin for alert triggers based on label representativity.
                Slack is linearly adjusted for n>2 classes.
        TODO: Plot bar chart with observation counts for each class and respective thresholds"""
        if slack < 0 or slack > 0.5:
            raise ValueError('Slack must be part of the open interval ]0, 0.5[')
        label_counts = self._get_label_counts(dropna=True)  # No label records are not considered
        n_classes = len(label_counts)
        labeled_records = label_counts.sum()  # Total labelled records
        label_ratio = label_counts/labeled_records
        fair_share = 1/n_classes
        adj_slack = slack*(2/n_classes)  # Adjust slack depending on number of classes
        label_excess = (label_ratio - fair_share)[abs(label_ratio - fair_share) > adj_slack]
        data = {}
        if len(label_excess) != 0:
            for _class, excess in label_excess.items():
                folder = 'Under-represented'  # By default
                if excess > 0:
                    folder = 'Over-represented'
                data.setdefault(folder, {})[_class] = self.df[self.df[self.label] == _class]
            self._warnings.add(
                QualityWarning(
                    test='Unbalanced Classes', category='Labels', priority=2,
                    data=data,
                    description="""Classes {} are under-represented each having less than {:.1%} of total instances.\
                        \nClasses {} are over-represented each having more than {:.1%} of total instances""".format(
                        set(data['Under-represented'].keys()), fair_share-adj_slack,
                        set(data['Over-represented'].keys()), fair_share+adj_slack)
            ))
        else:
            print("[UNBALANCED CLASSES] No unbalanced classes were found.")
            return None
        return label_excess.index

    def one_vs_rest_performance(self, slack: float = 0):
        """Performs one vs rest classification over each label class.
        Returns a series with Area Under Curve for each label class.
        Slack defines a proportion for the record weighted average of performances as a tolerance.
        Any binary classifier that scores below the average minus tolerance triggers a warning.
        TODO; Plot ROC curve"""
        assert 0<=slack<=1, "Argument th is expected to be a float in the [0,1] interval"
        _class_counts = self._get_label_counts(dropna=True)
        _class_counts = _class_counts[_class_counts>1]
        results = {
            _class: performance_one_vs_rest(df=self.tdf, label_feat=self.label,
                                            _class=_class, dtypes=self.dtypes)
            for _class in _class_counts.index
        }
        record_weighted_avg = sum([perf*_class_counts[_class] for _class, perf in results.items()])
        record_weighted_avg = (1/_class_counts.sum())*record_weighted_avg
        threshold = (1-slack)*record_weighted_avg
        poor_performers = {_class:perf for _class, perf in results.items() if perf < threshold}
        if len(poor_performers)>0:
            self._warnings.add(
                QualityWarning(
                    test='One vs Rest Performance', category='Labels', priority=2,
                    data=pd.Series(poor_performers),
                    description="Classes {} performed under the {:.1%} AUROC threshold.\
                    \n\tThe threshold was defined as an average of all classifiers with {:.0%} slack.".format(
                            set(poor_performers.keys()), threshold, slack
                        )
            ))
        return pd.Series(results)

    def _get_centroids(self):
        """Produces a centroid estimation for observations grouped by label value.
        Centroids are estimated using the normalized dataset."""
        label_counts = self._get_label_counts(dropna=True)
        centroids = pd.DataFrame(self.tdf.iloc[:len(label_counts)],
                                columns=self.tdf.columns, index=label_counts.index)
        for i, _class in enumerate(label_counts.index):
            records = self.tdf[self.tdf[self.label]==_class]
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
            records = self.tdf[self.tdf[self.label]==_class].drop(self.label, axis=1)
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
            sd_distances = sds['Scaled Distances'][sds['Scaled Distances']>th]
            new_outliers = len(sd_distances)
            if new_outliers > 0:
                potential_outliers += new_outliers
                data.setdefault(_class, self.df.loc[sd_distances.index])
        if potential_outliers>0:
            self._warnings.add(
                QualityWarning(
                    test='Outlier Detection', category='Labels', priority=2, data=data,
                    description="Found {} potential outliers across {} classes.\
                        \n\tA distance bigger than {} standard deviations of intra-cluster distances to the respective centroids was used to define the potential outliers.".format(
                                potential_outliers, len(data.keys()), th)
            ))
        return data


class NumericalLabelInspector(SharedLabelInspector):
    """Engine for running analyis on labels.
    Ordinal labels can be handled if passed as categorical.
    Numerical labels are not currently supported."""

    def __init__(self, df: pd.DataFrame, label: str):
        super().__init__(df, label)
        self._tests = ["missing_labels", "test_normality", "outlier_detection"]

    def _equal_clusters(self, nbins: int =5):
        """Sorts by label and separates the dataset into nbins of equal size
        TODO: Applications based on this? Initially it was meant for a bin width comparison. Considering deprecating"""
        sorted_vals = self.tdf[self.label].sort_values().copy()
        n_rows = len(sorted_vals)
        avg_bin_size = n_rows/nbins
        sorted_vals[:] = list(range(n_rows))
        return sorted_vals.apply(lambda x: int(x/avg_bin_size))

    def _KMeans_clusters(self, max_clusters):
        """Separates the dataset into a KMeans cluster optimized nbins.
        Clustering is done only with the label column values.
        TODO: Considering deprecating, full cluster usability currently with GMM"""
        sorted_vals =  self.tdf[self.label].sort_values().copy()
        search_space = range(1, max_clusters)
        inertias = [None for k in search_space]
        labels = {k: None for k in search_space}
        for i, k in enumerate(search_space):
            labels[k], inertias[i] = kmeans(sorted_vals.values.reshape(-1, 1), k)
        ideal_k = elbow(search_space, inertias)
        return pd.Series(labels[ideal_k], index=sorted_vals.index)

    def _GMM_clusters(self, max_clusters):
        """Separates the dataset into a Gaussian Mixture Model cluster optimized nbins.
        Clustering is done only with the label column values."""
        sorted_vals =  self.tdf[self.label].sort_values().copy()
        search_space = range(1, max_clusters)
        AICs = [None for k in search_space]
        labels = {k: None for k in search_space}
        for i, k in enumerate(search_space):
            labels[k], AICs[i] = GMM_clustering(sorted_vals.values.reshape(-1, 1), k)
        ideal_k = list(labels.keys())[AICs.index(min(AICs))]
        return pd.Series(labels[ideal_k], index=sorted_vals.index)

    def outlier_detection(self, th: float =3., use_clusters=False, max_clusters: int = 5):
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
            cluster_labels = pd.Series('full_dataset', index=self.tdf.index)
        clusters = cluster_labels.unique()
        potential_outliers = {}
        for cluster in clusters:
            values = self.tdf[self.label][cluster_labels==cluster].copy()
            if len(values) == 1:  # Single element clusters are automatically flagged as potential outliers
                potential_outliers[cluster] = self.df.loc[values.index]
            else:  # Look for deviant elements inside clusters
                median = values.median()
                std = values.std()
                abs_deviations = ((values-median)/std).abs()
                cluster_outliers = self.df.loc[abs_deviations[abs_deviations>th].index]
                if len(cluster_outliers)>0:
                    potential_outliers[cluster] = cluster_outliers
        if len(potential_outliers)>0:
            total_outliers = sum([cluster_outliers.shape[0] for cluster_outliers in potential_outliers.values()])
            coverage_string = "{} clusters".format(len(clusters)) if use_clusters else "the full dataset"
            self._warnings.add(
                QualityWarning(
                    test='Outlier Detection', category='Labels', priority=2, data=potential_outliers,
                    description="Found {} potential outliers across {}.\
                        \n\tA distance bigger than {} standard deviations of intra-cluster distances to the respective centroids was used to define the potential outliers.".format(
                                total_outliers, coverage_string, th)
            ))
        return potential_outliers

    def test_normality(self, p_th=5e-3):
        """Runs a normal distribution test on the label column.
        If passes data is normally distributed.
        If it fails, retries with a battery of transformations.
        """
        vals =  self.tdf[self.label].copy()
        test_result, transform, pstat = normality_test(vals, p_th=p_th)
        if test_result:
            if transform is None:
                print("[TEST NORMALITY] The label values appears to be normally distributed.")
            else:
                print("[TEST NORMALITY] The {} transform appears to be able to normalize the label values.".format(transform))
                self._warnings.add(
                    QualityWarning(
                        test='Test normality', category='Labels', priority=2, data=vals,
                        description="The label distribution as-is failed a normality test.\
                            \n\tUsing the {} transform provided a positive normality test with a p-value statistic of {:.2f}".format(
                                    transform, pstat)
                ))
        else:
            print("[TEST NORMALITY] It was not possible to normalize the label values. See the warning message for additional context.")
            self._warnings.add(
                QualityWarning(
                    test='Test normality', category='Labels', priority=1, data=vals,
                    description="The label distribution failed to pass a normality test as-is and following a battery of transforms.\
                        \n\tIt is possible that the data originates from an exotic distribution, there is heavy outlier presence or it is multimodal.\
                        \n\tAddressing this issue might prove critical for regressor performance."
            ))
