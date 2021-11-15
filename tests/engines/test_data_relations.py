"Tests for the DataRelations module."
from pytest import fixture
from pandas import read_csv
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from ydata_quality.data_relations.engine import DataRelationsDetector


@fixture
def data_relations():
    return DataRelationsDetector()

@fixture
def example_dataset_transformed():
    dataset_path = 'datasets/transformed/census_10k.csv'
    return read_csv(dataset_path)

@fixture
def ipynb_tutorial():
    path = "tutorials/data_relations.ipynb"
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)
    return nb

@fixture
def dr_results_no_pcorr(data_relations, example_dataset_transformed):
    results = data_relations.evaluate(
                                      df = example_dataset_transformed,
                                      dtypes = None,
                                      label = 'income',
                                      plot = False)
    return data_relations, results

@fixture
def dr_results_pc_corr(data_relations, example_dataset_transformed):
    df = example_dataset_transformed.drop(columns=['education-num'])
    results = data_relations.evaluate(
                                      df = df,
                                      dtypes = None,
                                      label = 'income',
                                      plot = False)
    return data_relations, results

def test_get_warnings(dr_results_no_pcorr):
    new_drd = DataRelationsDetector()
    assert isinstance(new_drd.get_warnings(), list)
    assert len(new_drd.get_warnings()) == 0

    ran_data_relations, _ = dr_results_no_pcorr
    assert isinstance(ran_data_relations.get_warnings(), list)
    assert len(ran_data_relations.get_warnings()) > 0

def test_results(dr_results_no_pcorr, dr_results_pc_corr):
    _, results = dr_results_no_pcorr
    assert isinstance(results, dict)
    assert set(results.keys()) == set(['Correlations', 'Feature Importance', 'High Collinearity'])

    _, results2 = dr_results_pc_corr
    assert isinstance(results2, dict)
    assert set(results2.keys()) == set(['Correlations', 'Confounders', 'Colliders', 'Feature Importance', 'High Collinearity'])

def test_tutorial_notebook_execution(ipynb_tutorial):
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    assert ep.preprocess(ipynb_tutorial, {'metadata': {'path': "tutorials"}})
