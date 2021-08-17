"""
Implementation of DataExpectationsReporter engine to run data expectations validation analysis.
"""
import json

from typing import Optional, Union

import numpy as np
import pandas as pd

from ydata_quality.core import QualityEngine, QualityWarning


class DataExpectationsReporter(QualityEngine):
    """Main class to run data expectation validation analysis.
    Supports standard Great Expectations json reports from expectation suite validation runs.
    """

    @property
    def tests(self):
        "List of individual tests available for the data quality checks."
        return ['_overall_assessment', '_coverage_fraction', '_expectation_level_assessment']

    @staticmethod
    def __test_json(results_json: Union[str, dict]):
        """Tests type of results_json.
        If a string is passed, tests file existance from given path and attempts to parse as a json file.
        Else accepts a Python dictionary (loaded json file).

        Args:
          results_json (Union[str, dict]): A path to the Great Expectations validation run output or its loaded version.
        """
        if isinstance(results_json, str):
            with open(results_json, 'r') as b_stream:
                data = b_stream.read()
            results_json = json.loads(data)
        elif not isinstance(results_json, dict):
            raise IOError("Expected a path to json file or a parsed json file (Python Dictionary) of a Great\
 Expectations validation run results.")
        return results_json

    @staticmethod
    def __test_df(df: pd.DataFrame):
        """Tests type of the provided DataFrame (if provided).
        Returns None if test fails.

        df (pd.DataFrame): The Pandas DataFrame that ran against the expectation suite."""
        try:
            assert isinstance(df, pd.DataFrame), "Currently only Pandas DataFrames are supported in the df argument."
            return df
        except:
            return None

    def __between_value_error(self, expectation_summary):
        """Computes deviation metrics of the observed value relative to the expectation range and the nearest bound.
        If the max and the min of the range coincide, deviation_relative_to_range is returned None.
        If the nearest bound is 0, deviation_relative_to_bound is not computed is returned None.
        Returns a signed tuple (deviation_relative_to_bound, deviation_relative_to_range)."""
        range_deviations, bound_deviations = (None, None)
        observed = expectation_summary['result']['observed_value']
        bounds = [expectation_summary['kwargs'][bound] for bound in ['min_value', 'max_value']]
        abs_dist_bounds = [abs(observed-bound) for bound in bounds]
        nearest_bound = bounds[np.argmin(abs_dist_bounds)]
        range_width = bounds[1]-bounds[0]
        deviation = observed - nearest_bound
        if range_width != 0:
            range_deviations = deviation/range_width
        if nearest_bound != 0:
            bound_deviations = deviation/nearest_bound
        range_deviation_string = "\n\t- The observed deviation is of {:.1f} min-max ranges.".format(range_deviations)
        bound_deviation_string = "\n\t- The observed value is {:.0%} deviated from the nearest bound of the expected\
 range.".format(bound_deviations)
        self.store_warning(
            QualityWarning(
                test='Expectation assessment - Value Between', category='Data Expectations', priority=3, data=(range_deviations, bound_deviations),
                description="The observed value is outside of the expected range."
                +(range_deviation_string if range_deviation_string else "")+(bound_deviation_string if bound_deviation_string else "")
            )
        )
        return (range_deviations, bound_deviations)

    def _summarize_results(self, results_json: Union[str, dict]):
        """Tests and parses the results_json file, creates a metadata summary to support tests of the module.

        Args:
            results_json (Union[str, dict]): A already summarized version of the results log, a path to the Great
                Expectations validation run output or its loaded version."""
        results_json = self.__test_json(results_json)
        if '__is_summary' in results_json:  # Indicates that the provided results_json is already a summarized version
            results_summary = results_json
        else:
            results_summary = {'OVERALL': {},
            'EXPECTATIONS': {}}
            for idx_, expectation_results in enumerate(results_json["results"]):
                expectation_summary = {
                    'results_format': "BASIC+" if "result" in expectation_results else "BOOLEAN_ONLY",
                    "success": expectation_results['success'],
                    "type": expectation_results['expectation_config']['expectation_type'],
                    "kwargs": expectation_results['expectation_config']['kwargs'],
                    "result": expectation_results['result']
                }
                expectation_summary['is_table_expectation'] = expectation_summary['type'].startswith("expect_table_")
                expectation_summary['column_kwargs'] = {k:v for k, v in expectation_results['expectation_config']['kwargs'] if k.startswith('column')}
                results_summary["EXPECTATIONS"][idx_] = expectation_summary
            overall_results = {'expectation_count': len(results_summary["EXPECTATIONS"]),
            "total_successes": sum([True for summary in results_summary["EXPECTATIONS"].values() if summary['success']])}
            overall_results["success_rate"] = overall_results["total_successes"]/overall_results["expectation_count"]
            results_summary['OVERALL'] = overall_results
        return results_summary

    def _coverage_fraction(self, results_json: Union[str, dict], df: pd.DataFrame, expected_minimum_coverage=0.75):
        """Compares the DataFrame column schema to the results json file to estimate validation coverage fraction.
        Ignores all table expectations (since these either are not comparing columns or are catchall expectations).

        Args:
            results_json (Union[str, dict]): A already summarized version of the results log, a path to the Great
                Expectations validation run output or its loaded version.
            expected_minimum_coverage (float): A fraction of the minimum DataFrame columns expected coverage by the expectation suite.
            df (pd.DataFrame): The Pandas DataFrame that ran against the expectation suite, used to evaluate coverage."""
        results_summary = self._summarize_results(results_json)
        df_column_set = set(df.columns())
        column_coverage = set()
        for summary in results_summary['EXPECTATIONS'].values():
            if summary['is_table_expectation']:
                continue  # Table expectations are not considered
            for kwarg in summary['column_kwargs'].values():
                if isinstance(kwarg, str):
                    kwarg = [kwarg]
                column_coverage.update(kwarg)
        assert column_coverage.issubset(df_column_set), "The column coverage of the validation run appears to originate from a different DataFrame."
        coverage_fraction = len(column_coverage)/len(df_column_set)
        if coverage_fraction < expected_minimum_coverage:
            self.store_warning(
                    QualityWarning(
                        test='Coverage Fraction', category='Data Expectations', priority=2, data=df_column_set.difference(column_coverage),
                        description="The provided DataFrame has a total expectation coverage of {:.0%} of its columns, which is below the expected coverage of {:.0%}.".format(
                            coverage_fraction, expected_minimum_coverage)
                    )
                )
        return len(column_coverage)/len(df_column_set)

    def _overall_assessment(self, results_json: Union[str, dict], error_tol: int = 0, rel_error_tol: Optional[float] = None):
        """Controls for errors in the overall execution of the validation suite.
        Raises a warning if failed expectations are over the tolerance (0 by default).

        Args:
            results_json (Union[str, dict]): A already summarized version of the results log, a path to the Great
                Expectations validation run output or its loaded version.
            error_tol (int): Defines how many failed expectations are tolerated.
            rel_error_tol (float): Defines the maximum fraction of failed expectations, overrides error_tol."""
        results_summary = self._summarize_results(results_json)
        overall_results = results_summary['OVERALL']
        failed_expectation_idxs = [i for i, expectation in enumerate(results_json['EXPECTATIONS']) if not expectation['success']]
        if rel_error_tol:
            error_tol = overall_results['expectation_count']*rel_error_tol
        if overall_results['expectation_count'] - overall_results['total_successes'] > error_tol:
            self.store_warning(
                QualityWarning(
                    test='Overall Assessment', category='Data Expectations', priority=2, data=failed_expectation_idxs,
                    description="{} expectations have failed, which is more than the implied absolute threshold of {} failed expectations.".format(
                        len(failed_expectation_idxs), int(error_tol))
                )
            )
        return failed_expectation_idxs

    def _expectation_level_assessment(self, results_json):
        """Controls for errors in the expectation level of the validation suite.
        Calls expectation specific methods to analyze some of the expectation logs.

        Args:
            results_json (Union[str, dict]): A already summarized version of the results log, a path to the Great
                Expectations validation run output or its loaded version."""
        results_summary = self._summarize_results(results_json)
        expectation_level_report = pd.DataFrame(index = results_summary['EXPECTATIONS'].keys(), columns =
        ['Expectation type', 'Result', 'Error metric(s)'])
        for idx_, expectation_summary in results_summary['EXPECTATIONS'].items():
            error_metric = None
            result = expectation_summary["success"]
            expectation_type = expectation_summary["type"]
            if result is False:
                # Expectation specific rules are called here
                if "between" in expectation_type and "quantile" not in expectation_type:
                    error_metric = self.__between_value_error(expectation_summary)
            expectation_level_report.iloc[idx_] = [expectation_type, result, error_metric]
        return expectation_level_report

    def evaluate(self, results_json, df = None, error_tol = 0, rel_error_tol: Optional[float] = None, expected_minimum_coverage: Optional[float] = 0.75):
        """Runs tests to the validation run results and reports based on found errors.

        Args:
            results_json (Union[str, dict]): A already summarized version of the results log, a path to the Great
                Expectations validation run output or its loaded version.
            df (pd.DataFrame): The Pandas DataFrame that ran against the expectation suite, used to evaluate coverage.
            error_tol (int): Defines how many failed expectations are tolerated.
            rel_error_tol (float): Defines the maximum fraction of failed expectations, overrides error_tol.
            expected_minimum_coverage (float): A fraction of the minimum DataFrame columns expected coverage by the expectation suite."""
        self._warnings = set() # reset the warnings to avoid duplicates
        df = self.__test_df(df)
        results_summary = self._summarize_results(results_json)
        results = {}
        results['Overall Assessment'] = self._overall_assessment(results_summary, error_tol, rel_error_tol)
        if df:
            results['Coverage Fraction'] = self._coverage_fraction(results_summary, df, expected_minimum_coverage=expected_minimum_coverage)
        else:
            print("A valid DataFrame was not passed, skipping coverage fraction test.")
        results['Expectation level assessment'] = self._expectation_level_assessment(results_summary)
        return results
        