"""
Implementation of DriftAnalyser engine to run data drift analysis.
"""
from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.stats._continuous_distns import chi2_gen
from ydata_quality.core import QualityEngine, QualityWarning


class ModelWrapper:
    """Base class for model wrapper.
    Defines a Model instance to enable concept drift analysis with the Sampling engine.
    This class is meant to cover all functionality needed to interact with the engine.
    Can be instantiated directly or inherited from a custom class.
    In normal use only the preprocess and postprocess methods should need override."""

    def __init__(self, model: Callable):
        self._model = model

    @property
    def model(self):
        """Passes the provided callable as the property model."""
        return self._model

    @staticmethod
    def _preprocess(x: pd.DataFrame):
        """Performs any preprocessing of the model input.
        By default returns input without any transformation.
        Override to define custom preprocessing steps."""
        return x

    @staticmethod
    def _postprocess(y: pd.Series):
        """Performs any postprocessing of the models label predictions.
        By default returns input without any transformation.
        Override to define custom model predictions postprocessing steps."""
        return y

    def _predict(self, x: pd.DataFrame):
        """Runs the provided callable model on pretransformed input."""
        if hasattr(self.model, "predict"):  # Sklearn and tensorflow model standards
            return self.model.predict(x)
        else:  # Pytorch and other __call__ prediction standards
            return self.model(x)

    def __call__(self, x: pd.DataFrame) -> pd.Series:
        """Returns a sample of labels predicted by the model from the covariate sample x.
        The returned Series is expected to have the same number of rows as x."""
        transformed_x = self._preprocess(x)
        raw_y = self._predict(transformed_x)
        return self._postprocess(raw_y)


class DriftAnalyser(QualityEngine):
    """Main class to run drift test analysis.

    Methods:
        ref_covariate_drift: controls covariate drift in reference subsamples.
        ref_label_drift: controls label drift in the reference subsamples.
        sample_covariate_drift: detects covariate drift in the test sample, measured against the full reference sample.
        sample_label_drift: detects label drift in the test sample, measured against the full reference sample.
        sample_concept_drift: detects concept drift in the test sample based on a wrapped model provided by the user.
    """

    def __init__(self, ref: pd.DataFrame, sample: Optional[pd.DataFrame] = None,
        label: Optional[str] = None, model: Callable = None, holdout_size: float = 0.2):
        """
        Initializes the engine properties and lists tests for automated evaluation.
        Args:
            ref (pd.DataFrame): reference sample used to run sampling analysis, ideally the users dataset or a train dataset.
            sample (Optional, pd.DataFrame): sample to test drift against the reference sample, can be new data, a slice of the train dataset or a test sample.
            label (Optional, str): defines a feature in the provided samples as label.
            model (Optional, ModelWrapper): a custom model wrapped by the ModelWrapper class. The model is expected to perform label prediction over the set of features (covariates) of the provided samples.
            holdout_size (float): Fraction to be kept as holdout for drift test.
        """
        super().__init__(df=ref, label=label)
        self.sample = sample
        self._model = model
        self.has_model = None
        self._holdout, self._leftover = self._random_split(ref, holdout_size)
        self._tests = ['ref_covariate_drift', 'ref_label_drift', 'sample_covariate_drift',
            'sample_label_drift', 'sample_concept_drift']

    @property
    def sample(self):
        "Returns the user provided test sample."
        return self._sample

    @sample.setter
    def sample(self, sample: pd.DataFrame):
        if sample is not None:
            assert sorted(list(sample.columns)) == sorted(list(self.df.columns)), "The reference and independent samples must share schema."
        self._sample = sample

    @property
    def has_model(self):
        return self._has_model

    @has_model.setter
    def has_model(self, _):
        try:
            self._has_model = self.__test_model()
        except AssertionError:
            print("The provided model failed to produce output in the expected format during test and will not be used by the engine.")
            self._has_model = False
        except:
            self._has_model = False

    def __test_model(self):
        """Tests the provided model wrapper.
        Creates an example input from the provided samples.
        A valid test output is a label series with the same number of rows as x.
        Raises AssertionError if the model test fails.
        Raises a general exception if the conditions for test were not met.
        Please remove eventual label column from test_x before passing it to the test."""
        if self.label and self._model is not None:
            test_x = self.df.head().copy()
            test_x.drop(self.label, axis=1, inplace=True)
            output = self._model(test_x)
            assert isinstance(output, (pd.Series, np.ndarray)), "The provided model failed to produce the expected output."
            assert len(output) == test_x.shape[0], "The provided model failed to produce output with the expected dimensionality."
            return True
        raise Exception

    @staticmethod
    def _random_split(sample: Union[pd.DataFrame, pd.Series], split_size: float, shuffle=True):
        """Shuffles sample and splits it into 2 partitions according to split_size.
        Returns a tuple with the split first (partition corresponding to split_size, and remaining second).
        Args:
            sample (pd.DataFrame): A sample to be split
            split_size (float): Fraction of the sample to be taken split
            shuffle (bool): If True shuffles sample rows before splitting"""
        assert 0<= split_size <=1, 'split_size must be a fraction, i.e. a float in the [0,1] interval.'
        if shuffle:
            sample = sample.sample(frac=1)  # Shuffle dataset rows
        split_len = int(sample.shape[0]*split_size)
        split = sample.iloc[:split_len]
        remainder = sample.iloc[split_len:]
        return split, remainder

    @staticmethod
    def _chisq_2samp(reference_data: pd.Series, test_data: pd.Series):
        """Asserts validity of performing chisquared test on two samples.
        Tests the hypothesis that the test_sample follows ref_sample's distribution.
        Will raise an AssertionError in case the test is not valid.
        Args:
            reference_data (pd.Series): Reference data, used to compute degrees of freedom and expectation
            test_data (pd.Series): Test data, compared to the reference data
        Returns:
            chi_stat (float): The chi squared statistic of this test
            p_val (float): The p-value of the tested hypothesis
        """
        ref_unique_freqs = reference_data.value_counts(normalize=True)
        test_unique_counts = test_data.value_counts()
        assert set(test_unique_counts.index).issubset(set(ref_unique_freqs.index)),"test_sample contains categories unknown to the ref_sample"
        test_expected_counts = ref_unique_freqs*len(test_data)
        assert sum(test_expected_counts<5)==0, "The test sample has categories with expected count below 5 (this sample is too small for chi-squared test)"
        chi_stat = sum(((test_unique_counts-test_expected_counts)**2)/test_expected_counts)
        p_val = 1-chi2_gen().cdf(x=chi_stat, df=len(ref_unique_freqs-1))
        return chi_stat, p_val

    def _2sample_feat_goof(self, ref_sample: pd.Series, test_sample: pd.Series):
        """Performs a goodness of fit test between 2 samples.
        The column dtype of the samples allows for an appropriate statistic test selection.
        Returns tuple (statistic_value, p_value, test_name).
        If the statistic test raises an exception, (-1, None, test_name) is returned instead.
        Args:
            ref_sample (pd.Series): Reference sample (Relevant distinction for chi-squared test)
            test_sample (pd.Series): Test sample"""
        statistics = {'categorical': ('Chi-Squared', self._chisq_2samp),
            'numerical': ('Kolmogorov-Smirnov', ks_2samp)}
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
        bonferroni_p = p_thresh/len(covariates.columns)  # Bonferroni correction
        all_p_vals = pd.DataFrame(index=perc_index, columns=covariates.columns)
        for i, fraction in enumerate(leftover_fractions):
            downsample, _ = self._random_split(covariates, fraction)
            p_vals = []
            for column in covariates.columns:
                _, p_val, _ = self._2sample_feat_goof(ref_sample = downsample[column],
                    test_sample = holdout[column])
                p_vals.append(p_val)
            all_p_vals.iloc[i] = p_vals
            control_metric.iloc[i] = 100*len([p for p in p_vals if p > bonferroni_p])/len(p_vals)
        all_p_vals['Corrected p-value threshold'] = bonferroni_p
        control_metric.plot(title='Reference sample covariate features no drift(%)',
            xlabel='Percentage of remaining sample used',
            ylabel='Percentage of no drift features',
            ylim = (0, 104), style='.-')
        plt.show()
        return all_p_vals

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
            ylabel=f'{test_name} test p-value', style='.-')
        plt.show()

    def sample_covariate_drift(self, p_thresh: float= 0.05):
        """Detects covariate drift in the test sample (measured against the full reference sample).
        The p-value threshold is adjusted for the multivariate case via Bonferroni correction.
        Any p-value below the adjusted threshold indicates test sample drift, raising a warning.
        The dtypes are used to decide the test to be applied per column (chi squared or KS).
        Args:
            p_thresh (float): The p_threshold used for the test.
        """
        if self.sample is None:
            return "[SAMPLE LABEL DRIFT] To run sample covariate drift, a test sample must be provided. Test skipped."
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
        n_drifted_feats = sum(test_summary['Verdict']=='Drift')
        n_invalid_tests = sum(test_summary['Verdict']=='Invalid test')
        if n_drifted_feats>0:
            self.store_warning(
                QualityWarning(
                    test='Sample covariate drift', category='Sampling', priority=2, data=test_summary,
                    description=f"""{n_drifted_feats} features accused drift in the sample test. The covariates of the test sample do not appear to be representative of the reference sample."""
            ))
        elif n_invalid_tests>0:
            self.store_warning(
                QualityWarning(
                    test='Sample covariate drift', category='Sampling', priority=3, data=test_summary,
                    description=f"""There were {n_invalid_tests} invalid tests found. This is likely due to a small test sample size. The data summary should be analyzed before considering the test conclusive."""
            ))
        else:
            print("[SAMPLE COVARIATE DRIFT] Covariate drift was not detected in the test sample.")
        return test_summary

    def sample_label_drift(self, p_thresh: float= 0.05):
        """Detects label drift in the test sample (measured against the full reference sample).
        A p-value below the adjusted threshold indicates test sample drift, raising a warning.
        The label dtype is used to decide the test to be applied (chi squared or KS).
        Args:
            p_thresh (float): The p_threshold used for the test.
        """
        if self.sample is None or self.label is None or self.label not in self.sample.columns:
            return "[SAMPLE LABEL DRIFT] To run sample label drift, a test sample must be provided with the defined label column. Test skipped."
        labels = self.df[self.label].copy()
        test_sample = self.sample[self.label].copy()
        stat_val, p_val, test_name = self._2sample_feat_goof(ref_sample = labels,
            test_sample = test_sample)
        test_summary = pd.Series(data=[test_name, stat_val, p_val, None],
            index=['Statistic', 'Statistic Value', 'p-value', 'Verdict'])
        test_summary['Verdict'] = 'OK' if p_val > p_thresh else ('Drift' if p_val>= 0 else 'Invalid test')
        if test_summary['Verdict']=='Drift':
            self.store_warning(
                QualityWarning(
                    test='Sample label drift', category='Sampling', priority=2, data=test_summary,
                    description=f"""The label accused drift in the sample test with a p-test of {p_val}, which is under the threshold {p_thresh}. The label of the test sample does not appear to be representative of the reference sample."""
            ))
        elif test_summary['Verdict']=='Invalid test':
            self.store_warning(
                QualityWarning(
                    test='Sample label drift', category='Sampling', priority=3, data=test_summary,
                    description=f"""The test was invalid. This is likely due to a small test sample size."""
            ))
        else:
            print("[SAMPLE LABEL DRIFT] Label drift was not detected in the test sample.")
        return test_summary

    def sample_concept_drift(self, p_thresh: float= 0.05):
        """Detects concept drift in the test sample resorting to a user provided model wrapper.
        Results may not be conclusive without first testing if the test sample has label or covariate drift.
        A p-value below the adjusted threshold indicates test sample concept drift, raising a warning.
        The label dtype is used to decide the test to be applied (chi squared or KS).
        Args:
            p_thresh (float): The p_threshold used for the test.
        """
        if not self.has_model or self.sample is None:
            return "[CONCEPT DRIFT] To run concept drift, a valid model, a test sample and label column must be provided. Test skipped."
        ref_sample = self.df.copy()
        test_sample = self.sample.copy()
        ref_sample.drop(self.label, axis=1, inplace=True)
        test_sample.drop(self.label, axis=1, inplace=True)
        ref_preds = pd.Series(self._model(ref_sample), name=self.label)
        test_preds = pd.Series(self._model(test_sample), name=self.label)
        stat_val, p_val, test_name = self._2sample_feat_goof(ref_sample = ref_preds,
            test_sample = test_preds)
        test_summary = pd.Series(data=[test_name, stat_val, p_val, None],
            index=['Statistic', 'Statistic Value', 'p-value', 'Verdict'])
        test_summary['Verdict'] = 'OK' if p_val > p_thresh else ('Drift' if p_val>= 0 else 'Invalid test')
        if test_summary['Verdict']=='Drift':
            self.store_warning(
                QualityWarning(
                    test='Concept drift', category='Sampling', priority=2, data=test_summary,
                    description=f"""There was concept drift detected with a p-test of {p_val}, which is under the threshold {p_thresh}. The model's predicted labels for the test sample do not appear to be representative of the distribution of labels predicted for the reference sample."""
            ))
        elif test_summary['Verdict']=='Invalid test':
            self.store_warning(
                QualityWarning(
                    test='Concept drift', category='Sampling', priority=3, data=test_summary,
                    description=f"""The test was invalid. This is likely due to a small test sample size."""
            ))
        else:
            print("[CONCEPT DRIFT] Concept drift was not detected between the reference and the test samples.")
        return test_summary
