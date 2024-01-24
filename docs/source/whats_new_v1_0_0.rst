.. _whats_new_v1_0_0:

===============
Version 1.0.0
===============

This release contains some major changes, including:
 
- Micro Focus is now OpenText. All the documentation containing copyright information has been updated to represent the new ownership of Vertica and its associated products.

- Requirements update: Python version 3.10-3.12 are now supported.
- Minimum supported Python version is 3.9.

.. note:: 
  
  An internal minimum function python decorator (@check_minimum_version) 
  warns users if any of  the Vertica modules do not meet the requirement for the function in use.

Versioning Guidelines
----------------------

We highly recommend transitioning to the syntax of version 1.0.0, as the previous versions were in beta and subject to potential changes at any moment. Our current versioning process follows a systematic approach:

 - **Major Version Increment:** This is reserved for substantial changes that involve a significant shift in syntax and may deprecate multiple elements. Users should be prepared for a notable adjustment in how they interact with the software.
 - **Minor Version Increment:** Occurs when introducing new functionalities and making substantial improvements. Users can anticipate enhanced features and capabilities without major disruptions to their existing workflows.
 - **Last Digit Increment:** Reserved for bug fixes and changes that do not influence the syntax or functionality of the previous version. These updates are aimed at enhancing stability and addressing issues without requiring users to adapt to a new syntax.

By adhering to this versioning strategy, users can effectively navigate updates, ensuring a smooth transition while benefiting from the latest features, improvements, and bug fixes.

Upcoming Changes: Deprecated Modules
-------------------------------------

Several modules have been deprecated as part of a code restructuring initiative. Please be aware that the following import will soon be unsupported:

.. code-block:: python
  
  # Moved to verticapy.sql.geo
  import verticapy.geo

  # Moved to verticapy.machine_learning
  # And also restructured
  import verticapy.learn

  # Moved to verticapy.sql.functions
  # And also restructured
  import verticapy.stats

  # Moved to verticapy.sdk.vertica
  import verticapy.udf

  # Moved to verticapy.sql
  # And also restructured
  import verticapy.utilities

  # Moved to verticapy.core
  # And also restructured
  import verticapy.vdataframe

  # New syntax: %load_ext verticapy.chart
  %load_ext verticapy.hchart

As this is a major version release and to uphold best practices, we are expeditiously phasing out all old methodologies. It is imperative to ensure a swift adaptation to the new syntax. Warnings have been issued for several imports, and they will soon be removed.

In parallel, we are actively developing a set of new tests. Consequently, the current 'tests' folder will soon be replaced by 'tests_new'. Your cooperation in transitioning to the updated syntax and directory structure is greatly appreciated.
  
Bug fixes
----------

- Adjusted-R squared now works with "k" parameter.
- Corrected calculation of Prevalence Threshold.
- Fixed nested pie plots.
- Improved ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.balance` method.
- load_model accepts feature names with parentheses.
- ``pandas_to_vertica`` method is renamed :py:func:`~verticapy.pandas.read_pandas` and it can now work with ``pandas.DataFrames`` that have a column full of ``NaN`` values.
- :py:class:`~verticapy.machine_learning.vertica.feature_extraction.text.TfidfVectorizer` replaces :py:class:`~verticapy.machine_learning.vertica.CountVectorizer`.
- AutoML Error: An error prompt is now displayed when no model fits.
- Cramer's V calculation is now fixed. See ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.corr`.
- Colors can now be changed correctly for Matplotlib Candlestick plot 
- :py:class:`~verticapy.machine_learning.vertica.IsolationForest` Anomaly plot is now fixed.
- Plotly :py:class:`~verticapy.machine_learning.vertica.LocalOutlierFactor` 3D plot is fixed.
- Graphviz tree plot display is fixed.

____

Machine Learning Support
-------------------------

- New Vertica algorithms supported:
  - :py:class:`~verticapy.machine_learning.vertica.IsolationForest` .
  - :py:class:`~verticapy.machine_learning.vertica.KPrototypes` .
  - :py:class:`~verticapy.machine_learning.vertica.PoissonRegressor` .
  - :py:class:`~verticapy.machine_learning.vertica.AR` .
  - :py:class:`~verticapy.machine_learning.vertica.MA` .
  - :py:class:`~verticapy.machine_learning.vertica.ARMA` .
  - :py:class:`~verticapy.machine_learning.vertica.ARIMA` .
  - :py:class:`~verticapy.machine_learning.vertica.feature_extraction.text.TfidfVectorizer` . It is still beta. 

- New method :py:meth:`~verticapy.machine_learning.vertica.XGBoostClassifier.features_importance`  for finding the feature importance for XGBoost models. 
- Classification metrics are now available for multiclass data/model using three methods: ``micro``, ``macro``, ``weighted``, ``score`` and ``none``.
  - ``average_precision_score`` is another new metric that is added to classification metrics.
  - ``roc_auc`` and ``prc_auc`` now work for multi-class classification using different averaging techniques stated above. 
- Model names are now optional
- Model Tracking and Versioning now supported.
  Check out :ref:`/notebooks/ml/model_tracking_versioning/index.ipynb` for more details.
- Model Export and Import:
  Now models can be exported to ``pmml``, ``tensorflow``, and ``binary``. They can now be exported to another User Defined Location.

.. note::
  
  For more information, see: :ref:`api.machine_learning` and :ref:`api.machine_learning.metrics` .

_____

SQL
-----

- ``vDataFramesSQL`` is deprecated. Now, :py:class:`~verticapy.vDataFrame` can be used directly to create :py:class:`~verticapy.vDataFrame` from SQL. For example:

.. code-block:: python

  import verticapy as vp
  vp.vDataFrame(
    "(SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2) x"
  )

The new format supports other methods for creating :py:class:`~verticapy.vDataFrame` .

.. code-block:: python

  vp.vDataFrame(
    {
      "X":[1,2,3],
      "Y":['a','b','c'],
    }
  )
  
_______

Plotting
---------

- Plotly is now the default plotting library, introducing improved visualizations. The Plotly plots are more interactive and enhance the user experience.
- Plotly Outliers plot now has the option to customize colors using the ``colors`` parameter.
- Plotly Voronoi plot colors can also be changed.
- Plotly LOF plot colors can be changed. 
- Validation Curve Plot now has the option to either return the curve or only display results.
- Fixed bounds for Highcharts ACF plot.
- For majority of plots, the colors can be changed by ``colors`` parameter.
- Added Plotly line plots: area, stacked, and fully-stacked.
- Plotly Contout plot colors can be modified.
- Plotly Range plot
  - Can draw multiple plots.
  - Color change is very easy with ``colors=[...]`` option e.g.

  .. code-block:: python

    fig = data.range_plot(
      ["col1", "col2"],
      ts = "date",
      plot_median = True,
      colors = ["black", "yellow"],
    )

- Plotly ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.scatter` plot now has the option to plot Bubble plot.
- Plotly Pie chart now has the option to change color and size.
- Highcharts Histogram plot is now available.
- Plotly Histogram plot now allows multiple plots.
- You can now easily switch between the plotting libraries using the following syntax:

  .. code-block:: python

    from verticapy import set_option

    set_option("plotting_lib","matplotlib")
    
.. note:: The ``hchart`` method of :py:class:`~verticapy.vDataFrame` is deprecated. The Highcharts plots can be plotted using the regular SQL plotting syntax by setting Highcharts as the default plotting library.

- The parameters ``custom_height`` and ``custom_width`` have been added to all plots so that the sizes can be changed as needed.

- Validators now ensure that only supported options are selected for the VerticaPy options.

- Users can now plot directly from SQL queries:

.. code-block:: python

  %load_ext verticapy.jupyter.extensions.chart_magic
  %chart -c sql_command -f input_file -k 'auto' -o output_file
  
The :py:func:`~verticapy.jupyter.extensions.chart_magic.chart_magic` command is similar to the hchart command, accepting four arguments:

  1. SQL command.
  2. Input file.
  3. Plot type (e.g. pie, bar, boxplot, etc.).
  4. Output file.

Example:

.. code-block:: python

  %chart -k pie -c "SELECT pclass, AVG(age) AS av_avg FROM titanic GROUP BY 1;"

Classification Metrics
-----------------------

Added support for many new classification and regression metrics.

The following metrics have been added to the :py:func:`~verticapy.machine_learning.metrics.classification_report` :
  - Akaike's Information Criterion (AIC).
  - Balanced Accuracy (BA).
  - False Discovery Rate (FDR).
  - Fowlkes-Mallows index.
  - Positive Likelihood Ratio.
  - Negative Likelihood Ratio.
  - Prevalence Threshold.
  - Specificity.

Most of the above metrics are new in this version and can be accessed directly.

The following metrics have been added to the :py:func:`~verticapy.machine_learning.metrics.regression_report` :
  - Mean Squared Log Error.
  - Quantile Error.

_____

Library Hierarchy
------------------

Import structures have changed. The code has been completely restructured, which means that going forward all imports will be done differently. Currently, we still allow the previous structure of import, but it will gradually be deprecated.

The new structure has the following parent folders:

- Core includes: ``vdataframe``, ``parsers``, ``string_sql``, and ``tablesample``.
- Machine Learning includes: ``model_selection``, ``metrics``, ``memmodels``, and also all the ML functions of Vertica (``vertica`` folder).
- SQL includes: ``geo``, ``dtypes``, ``insert``, ``drop``, all the SQL mathematical functions, etc.
- Jupyter includes: extensions such as magic ``sql`` and magic ``chart``.
- Datasets includes: ``loaders`` and sample datasets.
- Connection includes: ``connect``, ``read``, ``write``, etc.
- ``_config`` includes configurations.
- ``_utils`` includes all utilities.

.. note:: 
  
  The folders with "_" subscript are internal

For example, to use Vertica's :py:class:`~verticapy.machine_learning.vertica.LinearRegression`, it should now be imported as follows:

.. code-block:: python

  from verticapy.machine_learning.vertica import LinearRegression
  
To import the statistical test :py:func:`~verticapy.machine_learning.model_selection.statistical_tests.het_arch`:

.. code-block:: python

  from verticapy.machine_learning.model_selection.statistical_tests import het_arch
  
____

Added Model Tracking tool (MLOps)
----------------------------------
  
It is a common practice for data scientists to train tens of temporary models before picking one of them as their candidate model for going into production.

A model tracking tool can help each individual data scientist to easily track the models trained for an experiment (project) and compare their metrics for choosing the best one.

Example:

.. code-block:: python

  import verticapy.mlops.model_tracking as mt

  # creating an experiment
  experiment = mt.vExperiment(
      experiment_name = "multi_exp",
      test_relation = iris_vd,
      X = [
        "SepalLengthCm", 
        "SepalWidthCm", 
        "PetalLengthCm", 
        "PetalWidthCm",
      ],
      y = "Species",
      experiment_type = "multi",
      experiment_table = "multi_exp_table",
  )

  # adding models to the experiment after they are trained
  experiment.add_model(multi_model1)
  experiment.add_model(multi_model2)
  experiment.add_model(multi_model3)

  # listing models in the experiment
  experiment.list_models()
  # finding the best model in the experiment based on a metric
  best_model = experiment.load_best_model("weighted_precision")
  
- Added Model Versioning (MLOps)
  
  To integrate in-DB model versioning into VerticaPy, we added a new function, named "register", to the VerticaModel class. Calling this function will execute the register_model meta-function inside Vertica and registers the model. We also implemented a new class in VerticaPy, named RegisteredModel, in order to help a user with MLSUPERVISOR or DBADMIN privilege to work with the registered models inside the database.

  Example:

.. code-block:: python

  # training a model and then registering it

  model = RandomForestClassifier(name = "my_schema.rfc1")
  model.fit(
    "public.train_data",
    ["pred1", "pred2", "pred3"],
    "resp",
  )
  model.register("application_name")

  # for users with MLSUPERVISOR or DBADMIN privilege

  import verticapy.mlops.model_versioning as mv

  rm = mv.RegisteredModel("application_name")
  rm.change_status(version = 1, new_status = "staging")
  pred_vdf2 = rm.predict(new_data_vDF, version = 1)
  
Others
-------

- Docstrings have been enriched to add examples and other details that will help in creating a more helpful doc.
- A new dataset "Africa Education" (:py:func:`~verticapy.datasets.load_africa_education`) has been added to the dataset library. It can be easily imported using:

.. code-block:: python

  from verticapy.datasets import load_africa_education

- Now we use the ``DISTRIBUTED_SEEDED_RANDOM`` function instead of ``SEEDED_RANDOM`` in Vertica versions higher than 23.
- Some new functions that help in viewing and using nested data:
  - ``vDataFrame.``:py:meth:`~verticapy.vDataFrame.explode_array` is a :py:class:`~verticapy.vDataFrame` function that allows users to expand the contents of a nested column.
- Changes that do not affect the user experience include:
  - Code restructuring to improve readability and better collaboration using PEP8 standard.
  - Improved the code pylint score to 9+, which makes the code more professional and efficient.
  - Improved thorough Unit Tests that require considerably less time to compute, making the CI/CD pipeline more efficient.

- Verticapylab autoconnection. Slight modification to allow smooth integration of the upcoming VerticaPyLab.
  
Internal
=========

- Hints have been added to most functions to make sure the correct inputs are passed to all the functions.

- A python decorator ``@save_verticapy_logs`` is used to effectively log the usage statistics of all the functions.

- A set of common classes were created for effective collaboration and incorporation of other plotting libraries in the future.

- A new decorator ``@check_dtypes`` is used to ensure correct input for the functions.

- Updated the workflow to use the latest version of GitHub actions, and added a tox.ini file and the contributing folder.

- The new GitHub workflow now automatically checks for pylint score of the new code that is added. If the error score is below 10, then the tests fail.

- Added a check in the workflow for fomatting using black. If any files requires reformatting, the test fails and reports the relevant files.

  