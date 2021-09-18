from ydata_quality.core.warnings import QualityWarning
from ydata_quality.core.engine import QualityEngine
from ydata_quality.core.data_quality import DataQuality
import pandas as pd
import numpy as np


sample_warnings = [
    QualityWarning(
        test='Test 4', category='MODULE X', priority=0, data=[1,2,3],
        description="I'm just a warning message"),
    QualityWarning(
        test='Test 3', category='MODULE Y', priority=1, data=[1,2,3],
        description="I'm just a warning message"),
    QualityWarning(
        test='Test 2', category='MODULE Z', priority=2, data=[1,2,3],
        description="I'm just a warning message"),
    QualityWarning(
        test='Test 1', category='MODULE W', priority=3, data=[1,2,3],
        description="I'm just a warning message"),
    QualityWarning(
        test='Test 4', category='MODULE X', priority=3, data=[1,2,3],
        description="Another prio 3 warning"),
    QualityWarning(
        test='Test 1', category='MODULE W', priority=2, data=[1,2,3],
        description="Another prio 2 warning"),
    QualityWarning(
        test='Test 1', category='MODULE W', priority=1, data=[1,2,3],
        description="Another prio 1 warning"),
    QualityWarning(
        test='Test 1', category='MODULE W', priority=0, data=[1,2,3],
        description="Another prio 0 warning"),]


eng = QualityEngine(None)


for warning in sample_warnings:
    eng.store_warning(warning)

eng.report()

df = pd.read_csv('datasets/transformed/guerry_histdata.csv')
ED_EXTENSIONS = ['a_custom_EDV', 999999999, '!', '', 'UNKNOWN']
SENSITIVE_FEATURES = ['Suicides', 'Crime_parents', 'Infanticide']

dq = DataQuality(df=df, label='Pop1831', ed_extensions=ED_EXTENSIONS, results_json_path='datasets/original/taxi_long.json', sensitive_features=SENSITIVE_FEATURES, random_state=42)

full_results = dq.evaluate()

dq.report()
