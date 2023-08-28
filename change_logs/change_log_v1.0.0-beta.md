# Change log

 

This release contains some major changes, including:

 
- Micro Focus is now OpenText. All the documentation containing copyright information has been updated to represent the new ownership of Vertica and its associated products.


- Requirements update: Python version 3.8 is now required.

  Note: An internal minimum function python decorator (@check_minimum_version) warns users if any of the Vertica modules do not meet the requirement for the function in use.

  
- Bug fixes:
    - Adjusted-R squared now works with "K" parameter
    - Corrected calculation of Prevalence Threshold
    - Fixed nested pie plots
    - Improved vDataFrame.balance() function
    - load_model accepts feature names with parentheses
    - pandas_to_vertica() now works with dataframes that have a column full of NaN values
    - CountVectorizer supports larger datasets

- Machine Learning Support:
    - VerticaPy now supports the KPrototypes algorithm.
    - New function for finding the feature importance for XGBoost models.
    - Classification metrics are now available for multiclass data/model using three methods: micro, macro, and weighted.
      - average_precision_score is another new metric that is added to classification metrics.
      - roc_auc and prc_auc now work for multi-class classification using different averaging techniques stated above. 
    - Model names are now optional


- `vDataFramesSQL` is deprecated. Now, `vDataFrame` can be used directly to create `vDataFrame`s from SQL. For example:


  ```python
  import verticapy as vp
  vp.vDataFrame("(SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2) x")
  ```

  The new format supports other methods for creating `vDataFrame`s.

  ```python
  vp.vDataFrame({"X":[1,2,3],"Y":['a','b','c']})
  ```
 

- Plotting
    - Plotly is now the default plotting library, introducing improved visualizations. The Plotly plots are more interactive and enhance the user experience.
    - You can now easily switch between the plotting libraries using the following syntax:


    ```python
    from verticapy import set_option
    set_option("plotting_lib","matplotlib")
    ```

    Note that the "Hchart" function is deprecated. The Highcharts plots can be plotted using the regular SQL plotting syntax by setting Highcharts as the default plotting library.

    - The parameters "custom_height" and "custom_width" have been added to all plots so that the sizes can be changed as needed.

  
- Validators now ensure that only supported options are selected for the VerticaPy options.

 
- Users can now plot directly from SQL queries:

  ```python
  %load_ext verticapy.jupyter.extensions.chart_magic
  %chart -c sql_command -f input_file -k 'auto' -o output_file
  ```

  The chart command is similar to the hchart command, accepting four arguments:

  1. SQL command
  2. Input file
  3. Plot type (e.g. pie, bar, boxplot, etc.)
  4. Output file

  Example:

  ```python
  %chart -k pie -c "SELECT pclass, AVG(age) AS av_avg FROM titanic GROUP BY 1;"
  ```

- Added support for many new classification and regression metrics.

  The following metrics have been added to the classification report:
    - Akaike’s Information Criterion (AIC)
    - Balanced Accuracy (BA)
    - False Discovery Rate (FDR)
    - Fowlkes–Mallows index
    - Positive Likelihood Ratio
    - Negative Likelihood Ratio
    - Prevalence Threshold
    - Specificity

    Most of the above metrics are new in this version and can be accessed directly.

    The following metrics have been added to the regression report:
    - Mean Squared Log Error
    - Quantile Error

  
- Import structures have changed. The code has been completely restructured, which means that going forward all imports will be done differently. Currently, we still allow the previous structure of import, but it will gradually be deprecated.


  The new structure has the following parent folders:

   - Core [includes `vDataFrame`, parsers `string_sql`, and `tablesample`]
   - Machine Learning [includes model selection, metrics, memmodels, and also all the ML functions of Vertica]
   - SQL [includes dtypes, insert, drop, etc.]
   - Jupyter [includes extensions such as magic SQL and magic chart]
   - Datasets [includes loaders and sample datasets]
   - Connection [includes connect, read, write, etc.]
   - _config [includes configurations]
   - _utils [icnludes all utilities]

  *Note that the folders with "_" subscript are internal


  For example, to use Vertica's `LinearRegression`, it should now be imported as follows:

  ```python
  from verticapy.machine_learning.vertica import LinearRegression
  ```

  To import statistical tests:

  ```python
  from verticapy.machine_learning.model_selection.statistical_tests import het_arch
  ```

- Added Model Tracking tool (MLOps)
  It is a common practice for data scientists to train tens of temporary models before picking one of them as their candidate model for going into production.
A model tracking tool can help each individual data scientist to easily track the models trained for an experiment (project) and compare their metrics for choosing the best one.

Example:

  ```python
  import verticapy.mlops.model_tracking as mt

  # creating an experiment
  experiment = mt.vExperiment(
      experiment_name="multi_exp",
      test_relation=iris_vd,
      X=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
      y="Species",
      experiment_type="multi",
      experiment_table="multi_exp_table",
  )

  # adding models to the experiment after they are trained
  experiment.add_model(multi_model1)
  experiment.add_model(multi_model2)
  experiment.add_model(multi_model3)

  # list models in the experiment
  experiment.list_models()
  # findind the best model in the experiment based on a metric
  best_model = experiment.load_best_model("weighted_precision")
  ```
  
- Added Model Versioning (MLOps)
  To integrate model versioning into VerticaPy, we added a new function, named "register", to the VerticaModel class. Calling this function will execute the register_model meta-function inside Vertica and registers the model. We also implemented a new class in VerticaPy, named RegisteredModel, in order to help user with MLSUPERVISOR or DBADMIN privilege work with the registered models inside the database.

Example:

  ```python
  # training a model and then registering it
  model = RandomForestClassifier(name = "my_schema.rfc1")
  model.fit("public.train_data", ["pred1","pred2","pred3"], "resp")
  model.register("application_name")

  # for users with MLSUPERVISOR or DBADMIN privilege
  import verticapy.mlops.model_versioning as mv
  rm = mv.RegisteredModel("application_name")
  rm.change_status(version=1, new_status="staging")
  pred_vdf2 = rm.predict(new_data_vDF, version=1)
  ```
  
- Changes that do not affect the user experience include:

     - Code restructuring to improve readability and better collaboration using PEP8 standard.
     - Improved the code pylint score to 9+, which makes the code more professional and efficient.
     - Improved thorough Unit Tests that require considerably less time to compute, making the CI/CD pipeline more efficient.

 
- Verticapylab autoconnection. Slight modification to allow smooth integration of the upcoming VerticaPyLab.

  
## Internal


- Hints have been added to most functions to make sure the correct inputs are passed to all the functions.

- A python decorator (@save_verticapy_logs) is used to effectively log the usage statistics of all the functions.

- A set of common classes were created for effective collaboration and incorporation of other plotting libraries in the future.

- A new decorator (@check_dtypes) is used to ensure correct input for the functions.

- Updated the workflow to use the latest version of GitHub actions, and added a tox.ini file and the contributing folder.

- The new GitHub workflow now automatically checks for pylint score of the new code that is added. If the score is below 5, then the tests fail.

- Added a check in the workflow for fomatting using black. If any files requires reformatting, the test fails and reports the relevant files.

