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
    - Convenient switching of plotting libraries. You can now easily switch the plotting libraries using the following syntax:


    ```python
    from verticapy import set_option
    set_option("plotting_lib","matplotlib")
    ```

    Note that the function "Hchart" is also deprecated. The Highcharts plots can be plotted using the regular SQL plotting syntax mentioned below if Highcharts is set as the default the plotting library.

    - Custom width and height. The parameters "custom_height" and "custom_width" have been added to all plots so that the sizes can be changed as per the user requirement.

  
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

  New metrics added to classification report:
    - Akaike’s Information Criterion (AIC)
    - Balanced Accuracy (BA)
    - False Discovery Rate (FDR)
    - Fowlkes–Mallows index
    - Positive Likelihood Ratio
    - Negative Likelihood Ratio
    - Prevalence Threshold
    - Specificity

    Most of the above metrics are new in this version and can be accessed directly.

    New metrics added to the regression report:
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

- Updated the workflow to use the latest version of Github actions, and added a tox.ini file and a CONTRIBUTING.md file.

