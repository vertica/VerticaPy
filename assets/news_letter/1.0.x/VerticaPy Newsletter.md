# VerticaPy Newsletter

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/assets/img/logo.png' width="180px">
</p>

With the release of VerticaPy versions 1.0.0 and 1.0.1, we are thrilled to announce many new features and additions to VerticaPy! The below sections highlight what's new in each of these versions. For more information about these new features, including examples, check out the new and improved [VerticaPy documentation](https://www.vertica.com/python/documentation/1.0.x/html/index.html).

## VerticaPy 1.0.0

### Support
- Python versions 3.10-3.12 are now supported, with a minimum requirement of Python 3.9.
- Several modules have been deprecated. It is recommended to transition to the updated syntax. For more information, see the [documentation](https://www.vertica.com/python/documentation/1.0.x/html/whats_new_v1_0_0.html#upcoming-changes-deprecated-modules).

### Machine Learning Support
- We've added support for many Vertica algorithms, including
  - [Isolation Forest](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.vertica.ensemble.IsolationForest.html)
  - [KPrototypes](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.vertica.cluster.KPrototypes.html)
  - [PoissonRegressor](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.vertica.linear_model.PoissonRegressor.html)
  - [AR](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.vertica.tsa.AR.html)
  - [MA](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.vertica.tsa.MA.html)
  - [ARMA](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.vertica.tsa.ARMA.html)
  - [ARIMA](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.vertica.tsa.ARIMA.html)
  - [TfidfVectorizer](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.vertica.feature_extraction.text.TfidfVectorizer.html)
- With the newly supported ``XGBClassifier.features_importance()`` method, you can find the feature importance for XGBoost models. [Link](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.vertica.ensemble.XGBClassifier.features_importance.html)
- New [classification metrics](https://www.vertica.com/python/documentation/1.0.x/html/machine_learning_metrics.html#classification), which use various averaging techniques, are now available for multiclass data/models.
- We have integrated model versioning and model tracking into VerticaPy, allowing users to register and work with models inside the database. [Click here](https://www.vertica.com/python/documentation/1.0.x/html/machine_learning_ml_ops.html) for more information. 
- For better consistency, ``verticapy.machine_learning.model_selection.statistical_tests.seasonal_decompose`` now handles multiple variables using the ROW data type. [Link](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.machine_learning.model_selection.statistical_tests.tsa.seasonal_decompose.html)

### Bug Fixes
Various bug fixes, including adjustments to R squared, Prevalence Threshold, and improvements to several methods—such as ``vDataFrame.balance()``.

### Other
- A new dataset, "Africa Education", has been added to ``verticapy.datasets``. [Link](https://www.vertica.com/python/documentation/1.0.x/html/datasets.html)
- ``vDataFrame.SQL`` has been deprecated. Now, ``verticapy.vDataFrame`` can be used to directly create a vDataFrame from SQL queries.

For example:

```
import verticapy as vp
vp.vDataFrame(
"(SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2) x"
)
```

[Click here](https://www.vertica.com/python/documentation/1.0.x/html/datasets.html) for more information.

- Import structures have been updated. The code was completely restructured for better readability and collaboration.

## VerticaPy 1.0.1

Along with the above 1.0.0 improvements, VerticaPy version 1.0.1 includes the following updates:

### Options
The ``verticapy.set_option()`` function now allows you to set the following options:
- ``max_cellwidth``: Maximum width of VerticaPy table cells.
- ``max_tableheight``: Maximum height of VerticaPy tables.
- ``theme``: Set the display theme for VerticaPy objects to 'light' or 'dark'. 'dark' is recommended for night use, and 'light' is the default.

The default theme is "Light". It is recommended for daily use:

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/assets/img/light_theme.png' width="70%">
</p>

On the other hand, the "Dark" theme is suited for night-time use:

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/assets/img/dark_theme.png' width="70%">
</p>

For switching the themes, following syntax can be used:

```
import verticapy as vp
vp.set_option("theme", "dark")
```

### Diagnostics
The ``verticapy.performance.vertica.qprof.QueryProfiler`` class offers an extended set of functionalities, enabling the creation of complex trees with multiple metrics. This can help in finding ways to improve the performance of slow-running queries.

### Other
The docstrings throughout the documentation have been enriched with examples and further details, providing an improved doc experience.
