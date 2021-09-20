# ydata-quality

ydata_quality is an open-source python library for assessing Data Quality throughout the multiple stages of a data pipeline development. 

A holistic view of the data can only be captured through a look at data from multiple dimensions and `ydata_quality` evaluates it in a modular way wrapped into a single Data Quality engine. This repository contains the core python source scripts and walkthrough tutorials.

# Quickstart

The source code is currently hosted on GitHub at: https://github.com/ydataai/ydata-quality

Binary installers for the latest released version are available at the [Python Package Index (PyPI).](https://pypi.org/project/ydata-quality/)
```
pip install ydata-quality
```

### Comprehensive quality check in few lines of code

```python
from ydata_quality import DataQuality
import pandas as pd

#Load in the data
df = pd.read_csv('./datasets/transformed/census_10k.csv')

# create a DataQuality object from the main class that holds all quality modules
dq = DataQuality(df=df)

# run the tests
results = dq.evaluate()

# Output a report of the quality issues found by the engines
dq.report() 
```
```python
Warnings count by priority:
	Priority 1: 1 warning(s)
	Priority 2: 3 warning(s)
	TOTAL: 4 warning(s)
List of warnings sorted by priority:
	[DUPLICATE COLUMNS] Found 1 columns with exactly the same feature values as other columns. (Priority 1: heavy impact expected)
	[EXACT DUPLICATES] Found 3 instances with exact duplicate feature values. (Priority 2: usage allowed, limited human intelligibility)
	[FLATLINES] Found 4627 flatline events with a minimun length of 5 among the columns {'marital-status', 'workclass', 'income', 'native-country', 'capital-gain', 'capital-loss', 'education', 'occupation', 'workclass2', 'sex', 'education-num', 'hours-per-week', 'relationship', 'race'}. (Priority 2: usage allowed, limited human intelligibility)
	[PREDEFINED ERRONEOUS DATA] Found 1960 ED values in the dataset. (Priority 2: usage allowed, limited human intelligibility)
```
# Examples

Here you can find walkthrough tutorials and examples to familarize with different modules of `ydata_quality`

- [Start Here for Quick and Overall Walkthrough](https://github.com/ydataai/ydata-quality/blob/master/tutorials/main.ipynb)

To dive into any focussed module, and to understand how they work, here are tutorial notebooks:
1. [Bias and Fairness](https://github.com/ydataai/ydata-quality/blob/master/tutorials/bias_fairness.ipynb)
2.  [Data Expectations](https://github.com/ydataai/ydata-quality/blob/master/tutorials/data_expectations.ipynb)
3.  [Data Relations](https://github.com/ydataai/ydata-quality/blob/master/tutorials/data_relations.ipynb)
4.  [Drift Analysis](https://github.com/ydataai/ydata-quality/blob/master/tutorials/drift.ipynb)
5.  [Duplicates](https://github.com/ydataai/ydata-quality/blob/master/tutorials/duplicates.ipynb)
6.  Labelling: [Categoricals](https://github.com/ydataai/ydata-quality/blob/master/tutorials/labelling_categorical.ipynb) and [Numericals](https://github.com/ydataai/ydata-quality/blob/master/tutorials/labelling_numerical.ipynb)
7.  [Missings](https://github.com/ydataai/ydata-quality/blob/master/tutorials/missings.ipynb)
8.  [Erroneous Data](https://github.com/ydataai/ydata-quality/blob/master/tutorials/erroneous_data.ipynb)

# Contributing
We are open to collaboration! If you want to start contributing you only need to:
1. Search for an issue in which you would like to work. Issues for newcomers are labeled with good first issue.
2. Create a PR solving the issue.
3. We would review every PRs and either accept or ask for revisions.

You can also join the discussions on the #data-quality channel on our [Slack](http://slack.ydata.ai) and request features/bug fixes by opening issues on our repository.

# Support
For support in using this library, please join the #help Slack channel. The Slack community is very friendly and great about quickly answering questions about the use and development of the library. [Click here to join our Slack community!](http://slack.ydata.ai/)

# License
[GNU General Public License v3.0](https://github.com/ydataai/ydata-quality/blob/master/LICENSE)

