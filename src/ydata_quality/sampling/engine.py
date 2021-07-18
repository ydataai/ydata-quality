"""
Implementation of MissingProfiler engine to run missing value analysis.
"""
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.stats._continuous_distns import chi2_gen
from ydata_quality.core import QualityEngine, QualityWarning
from ydata_quality.utils.modelling import infer_dtypes


class SamplingAssistant(QualityEngine):
    "Main class to run sampling quality analysis."

    def __init__(self, ref: pd.DataFrame, sample: Optional[pd.DataFrame] = None,
        label: Optional[str] = None, holdout_size: float = 0.2):
        """
        Args:
            ref (pd.DataFrame): reference sample used to run sampling analysis
            sample (pd.DataFrame): an independent sample to test drift against the reference sample
            label (str): defines a feature in the provided samples as label
            holdout_size (float): Fraction to be kept as holdout for drift test
        """
        super().__init__(df=ref)
        self._dtypes = None
        self.sample = sample
        self._label = label
        self._holdout, self._leftover = self._random_split(ref, holdout_size)
        self._tests = ['ref_covariate_drift', 'ref_label_drift', 'sample_covariate_drift', 'sample_label_drift']

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
    def sample(self):
        "Property that returns the provided independent sample."
        return self._sample

    @sample.setter
    def sample(self, sample: pd.DataFrame):
        if sample is not None:
            assert list(sample.columns).sort() == list(self.df.columns).sort(), "The reference and independent samples must share schema."
        self._sample = sample

    @property
    def dtypes(self):
        "Infered dtypes for the dataset."
        if self._dtypes is None:
            self._dtypes = infer_dtypes(self.df)
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

    @staticmethod
    def _random_split(sample: Union[pd.DataFrame, pd.Series], split_size: float):
        """Shuffles sample and splits it into 2 partitions according to split_size.
        Returns a tuple with the split first (partition corresponding to split_size, and remaining second).
        Args:
            sample (pd.DataFrame)
            split_size (float): Fraction of the sample to be taken split"""
        assert 0<= split_size <=1, 'split_size must be a fraction, i.e. a float in the [0,1] interval.'
        sample_shuffled = sample.sample(frac=1)  # Shuffle dataset rows
        split_len = int(sample_shuffled.shape[0]*split_size)
        split = sample_shuffled.iloc[:split_len]
        remainder = sample_shuffled.iloc[split_len:]
        return split, remainder

    @staticmethod
    def _chisq_2samp(ref_sample: pd.Series, test_sample: pd.Series):
        """Asserts validity of performing chisquared test on test sample according to reference.
        Tests the hypothesis that the test_sample follows ref_sample's distribution.
        Will raise an AssertionError in case the test is not valid.
        Returns:
            chi_stat (float): The chi squared statistic of this test
            p_val (float): The p-value of the tested hypothesis
        """
        ref_unique_freqs = ref_sample.value_counts(normalize=True)
        test_unique_counts = test_sample.value_counts()
        assert set(test_unique_counts.index).issubset(set(ref_unique_freqs.index)),"test_sample contains categories unknown to the ref_sample"
        test_expected_counts = ref_unique_freqs*len(test_sample)
        assert sum(test_expected_counts<5)==0, "The test sample has categories with expected count below 5 (this sample is too small for chi-squared test)"
        chi_stat = sum(((test_unique_counts-test_expected_counts)**2)/test_expected_counts)
        p_val = 1-chi2_gen().cdf(x=chi_stat, df=len(ref_unique_freqs-1))
        return chi_stat, p_val

    @staticmethod
    def _ks_2samp(ref_sample: pd.Series, test_sample: pd.Series):
        return ks_2samp(ref_sample, test_sample)

    def _2sample_feat_goof(self, ref_sample: pd.Series, test_sample: pd.Series):
        """Performs a goodness of fit test between 2 samples.
        The column dtype of the sample and allows appropriate statistic test selection.
        Returns tuple (statistic_value, p_value, test_name).
        If the statistic test raises an exception, (-1, None, test_name) is returned instead.
        Args:
            ref_sample (pd.Series): Reference sample (Relevant distinction for chi-squared test)
            test_sample (pd.Series): Test sample"""
        statistics = {'categorical': ('Chi-Squared', self._chisq_2samp),
            'numerical': ('Kolmogorov-Smirnov', self._ks_2samp)}
        feat_dtype = self.dtypes[ref_sample.name]
        test_name, test = statistics[feat_dtype]
        try:
            statistic_value, p_value = test(ref_sample, test_sample)
        except:
            statistic_value, p_value = -1, None
        return statistic_value, p_value, test_name

    def ref_covariate_drift(self, p_thresh: float= 0.05):
        """Controls covariate drift in reference subsamples.
        The controlled metric is the number of features with no drift detection.
        This % is plotted against the size of the reference subsample.
        A monotonic increase of the value is expected as the subsample size is increased.
        The dtypes are used to decide the test to be applied per column (chi squared or KS).
        The p-value threshold is adjusted for the multivariate case via Bonferroni correction.
        Args:
            p_thresh (float): The p_threshold used for the test.
        """
        covariates = self._leftover.copy()
        holdout = self._holdout.copy()
        if self.label:
            covariates.drop(self.label, axis=1, inplace=True)
            holdout.drop(self.label, axis=1, inplace=True)
        leftover_fractions = np.arange(0.2, 1.2, 0.2)
        perc_index = ["{0:.0%}".format(fraction) for fraction in leftover_fractions]
        control_metric = pd.Series(index=perc_index)
        all_p_vals = pd.DataFrame(index=perc_index, columns=covariates.columns)
        bonferroni_p = p_thresh/len(covariates.columns)  # Bonferroni correction
        for i, fraction in enumerate(leftover_fractions):
            downsample, _ = self._random_split(covariates, fraction)
            p_vals = []
            for column in covariates.columns:
                _, p_val, _ = self._2sample_feat_goof(ref_sample = downsample[column],
                    test_sample = holdout[column])
                p_vals.append(p_val)
            all_p_vals.iloc[i] = p_vals
            control_metric.iloc[i] = 100*len([p for p in p_vals if p > bonferroni_p])/len(p_vals)
        control_metric.plot(title='Reference sample covariate features no drift(%)',
            xlabel='Percentage of remaining sample used',
            ylabel='Percentage of no drift features')
        all_p_vals['Corrected p-value threshold'] = bonferroni_p
        all_p_vals.plot(title='Reference sample covariate features test p_values',
            xlabel='Percentage of remaining sample used',
            ylabel='Test p-value',
            logy=True)
        plt.show()

    def ref_label_drift(self, p_thresh: float= 0.05):
        """Controls label drift in the reference sample (df).
        The p-value of the test is plotted against the size of the reference subsample.
        A monotonic increase of this metric is expected as we increase the subsample size.
        The dtype is used to decide the test to be applied to the label (chi squared or KS).
        Args:
            p_thresh (float): The p_threshold used for the test."""
        if self.label is None:
            return  "[REFERENCE LABEL DRIFT] No label was provided. Test skipped."
        labels = self._leftover[self.label].copy()
        holdout = self._holdout[self.label]
        leftover_fractions = np.arange(0.2, 1.2, 0.2)
        p_values = pd.DataFrame(index=["{0:.0%}".format(fraction) for fraction in leftover_fractions],
            columns=['Label p-value', 'p-value threshold'])
        for i, fraction in enumerate(leftover_fractions):
            downsample, _ = self._random_split(labels, fraction)
            _, p_val, test_name = self._2sample_feat_goof(ref_sample = downsample,
                test_sample = holdout)
            p_values['Label p-value'].iloc[i] = p_val
        p_values['p-value threshold'] = p_thresh
        p_values.plot(title='Reference sample label p-values',
            xlabel='Percentage of remaining sample used',
            ylabel=f'{test_name} test p-value')
        plt.show()

    def sample_covariate_drift(self, p_thresh: float= 0.05):
        """Detects covariate drift in the test sample (measured against the full reference sample).
        The p-value threshold is adjusted for the multivariate case via Bonferroni correction.
        Any p-value below the adjusted threshold indicates test sample drift, raising a warning.
        The dtypes are used to decide the test to be applied per column (chi squared or KS).
        Args:
            p_thresh (float): The p_threshold used for the test.
        """
        covariates = self.df.copy()
        test_sample = self.sample.copy()
        if self.label:
            covariates.drop(self.label, axis=1, inplace=True)
            test_sample.drop(self.label, axis=1, inplace=True)
        bonferroni_p = p_thresh/len(covariates.columns)  # Bonferroni correction
        test_summary = pd.DataFrame(index=covariates.columns,
            columns=['Statistic', 'Statistic Value', 'p-value', 'Verdict'])
        for column in covariates.columns:
            stat_val, p_val, test_name = self._2sample_feat_goof(ref_sample = covariates[column],
                test_sample = test_sample[column])
            test_summary.loc[column] = [test_name, stat_val, p_val, None]
        test_summary['Verdict'] = test_summary['p-value'].apply(
            lambda x: 'OK' if x > bonferroni_p else ('Drift' if x>= 0 else 'Invalid test'))
        drifted_feats = test_summary['Verdict']=='Drift'
        invalid_tests = test_summary['Verdict']=='Invalid test'
        if len(drifted_feats)>0:
            self._warnings.add(
                QualityWarning(
                    test='Sample covariate drift', category='Sampling', priority=2, data=test_summary,
                    description=f"""{drifted_feats.sum()} features accused drift in the sample test.\n
                    The covariates of the test sample do not appear to be representative of the reference sample."""
            ))
        elif len(invalid_tests)>0:
            self._warnings.add(
                QualityWarning(
                    test='Sample covariate drift', category='Sampling', priority=3, data=test_summary,
                    description=f"""There were {invalid_tests.sum()} invalid tests found. This is likely due to a small test sample size.\n
                    The data summary should be analyzed before considering the test conclusive."""
            ))
        else:
            print("[SAMPLE COVARIATE DRIFT] Covariate drift was not detected in the test sample.")
        return test_summary

    def sample_label_drift(self, p_thresh: float= 0.05):
        """Detecs label drift in the test sample (measured against the full reference sample).
        A p-value below the adjusted threshold indicates test sample drift, raising a warning.
        The label dtype is used to decide the test to be applied (chi squared or KS).
        Args:
            p_thresh (float): The p_threshold used for the test.
        """
        if self.label not in self.sample.columns:
            return  "[SAMPLE LABEL DRIFT] No label was provided in the test sample. Test skipped."
        labels = self.df[self.label].copy()
        test_sample = self.sample[self.label].copy()
        stat_val, p_val, test_name = self._2sample_feat_goof(ref_sample = labels,
            test_sample = test_sample)
        test_summary = pd.Series(data=[test_name, stat_val, p_val, None],
            index=['Statistic', 'Statistic Value', 'p-value', 'Verdict'])
        test_summary['Verdict'] = 'OK' if p_val > p_thresh else ('Drift' if p_val>= 0 else 'Invalid test')
        if test_summary['Verdict']=='Drift':
            self._warnings.add(
                QualityWarning(
                    test='Sample label drift', category='Sampling', priority=2, data=test_summary,
                    description=f"""The label accused drift in the sample test with a p-test of {p_val}, which is under the threshold {p_thresh}.\n
                    The label of the test sample does not appear to be representative of the reference sample."""
            ))
        elif test_summary['Verdict']=='Invalid test':
            self._warnings.add(
                QualityWarning(
                    test='Sample label drift', category='Sampling', priority=3, data=test_summary,
                    description=f"""The test was invalid. This is likely due to a small test sample size."""
            ))
        else:
            print("[SAMPLE LABEL DRIFT] Label drift was not detected in the test sample.")
        return test_summary
