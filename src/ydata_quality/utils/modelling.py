"""
Utilities based on building baseline machine learning models.
"""

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings

BASELINE_CLASSIFIER = Pipeline([
    ('imputer', SimpleImputer()),
    ('classifier', LogisticRegression())
])

BASELINE_REGRESSION = Pipeline([
    ('imputer', SimpleImputer()),
    ('classifier', LinearRegression())
])


@ignore_warnings(category=ConvergenceWarning)
def baseline_predictions(df: pd.DataFrame, target: str, type='classification'):
    "Train a baseline model and predict for a test set"

    # 1. Define the baseline model
    model = BASELINE_CLASSIFIER if type == 'classification' else BASELINE_REGRESSION

    # 2. Train overall model
    X, y = df.drop(target, axis=1), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train.select_dtypes('number'), y_train)

    # 3. Predict
    if type == 'regression':
        y_pred = model.predict(X_test.select_dtypes('number'))
    elif type == 'classification':
        y_pred = model.predict_proba(X_test.select_dtypes('number'))[:, 1]

    # 4. Return both the predictions and X_test, y_test to analyze the performances
    return y_pred, X_test, y_test

def performance_per_feature_values(df: pd.DataFrame, feature: str, target: str, type='classification'):
    """Performance achieved per each value of a groupby feature."""

    # 1. Define the baseline performance metric
    metric = roc_auc_score if type == 'classification' else mean_squared_error

    # 2. Get the baseline predictions
    y_pred, X_test, y_test = baseline_predictions(df=df, target=target, type=type)

    # 3. Get the performances per feature value
    uniques = set(X_test[feature])
    results =  {}
    for i in uniques: # for each category
        y_pred_cat = y_pred[X_test[feature]==i]
        y_true_cat = y_test[X_test[feature]==i]
        results[i] = metric(y_true_cat, y_pred_cat)
    return results

def performance_per_missing_value(df: pd.DataFrame, feature: str, target: str, type='classification'):
    """Performance difference between valued and missing values in feature."""

    # 1. Define the baseline performance metric
    metric = roc_auc_score if type == 'classification' else mean_squared_error

    # 2. Get the baseline predictions
    y_pred, X_test, y_test = baseline_predictions(df=df, target=target, type=type)

    # 3. Get the performance per valued vs missing feature
    missing_mask = X_test[feature].isna()
    results = {}
    results['missing'] = metric(y_test[missing_mask], y_pred[missing_mask])
    results['valued'] = metric(y_test[~missing_mask], y_pred[~missing_mask])
    return results

@ignore_warnings(category=ConvergenceWarning)
def predict_missingness(df: pd.DataFrame, feature: str):
    "Train a baseline model to predict the missingness of a feature value."
    # 0. Preprocessing
    df = df.copy() # avoid altering the original DataFrame
    target = f'is_missing_{feature}'

    # 1. Define the baseline model
    model = BASELINE_CLASSIFIER

    # 2. Create the new target
    df[target] = df[feature].isna()

    # 3. Train overall model
    X, y = df.drop([feature, target], axis=1), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train.select_dtypes('number'), y_train)

    # 4. Predict
    y_pred = model.predict_proba(X_test.select_dtypes('number'))[:, 1]

    # 5. Return the area under the roc curve
    return roc_auc_score(y_test, y_pred)

