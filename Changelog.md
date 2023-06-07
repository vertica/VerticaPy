# Change log

This release includes some major changes, including:

- Micro Focus is now Opentext

All the documentation containing copyright information have been updated to represent the new ownership of Vertica and its associated products. 

- Version number change

Instead of continuing the previous version style (0.12.0), the new version numbering will follow the OpenText version style. That means instead of 0.13.0 or 1.0.0, we will have 23.3 as the new version number. All the OpenText products follow the same version numbers for consistensy. 

- Requirements update

Now python version 3.8 is required.

Note: Now an internal minimum function python decorator (@check_minimum_version) warns the users if any of the Vertica modules does not meet the requirement for the function in use.

- Bug Fixes

-- Adjusted-R squared now works with "K" parameter
-- Calculation of Prevalence Threshold is corrected
-- Nested pie plots have been fixed
-- vDataFrame.balance() function is improved
-- load_model can now accept feature names with parentheses
-- pandas_to_vertica() works even if a dataframe has a column full of NaN values
-- CountVectorizer now works for larger datasets

- Machine Learning Support
  - KPrototypes is now supported in VerticaPy

- `vDataFramesSQL` is deprecated. Now `vDataFrame` can be used directly to create `vDataFrames` from SQL.

  New format:

  ```python
  import verticapy as vp
  vp.vDataFrame("(SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2) x")
  ```

  The new format also allows other ways to create a `vDataFrame`.

  ```python
  vp.vDataFrame({"X":[1,2,3],"Y":['a','b','c']})
  ```

- Plotly integration

  The new release comes with better visualization using Plotly as the default plotting library. Now the plots are more interactive to enhance the user experience.

- Convenient switching of plotting libraries

  The plotting libraries can now easily be switched using the following syntax:

  ```python
  from verticapy import set_option
  set_option("plotting_lib","matplotlib")
  ```

  Note that the function "Hchart" will also be deprecated as the Highcharts plots can be plotted using the regular SQL plotting syntax mentioned below while Highcharts is set as the default the plotting library.

- Validators for options

Validators now ensure that only allowed options are selected for the verticapy options. 

- Direct plotting from SQL queries

  In this update, users can directly plot from SQL queries.

  ```python
  %load_ext verticapy.jupyter.extensions.chart_magic
  %chart -c sql_command -f input_file -k 'auto' -o output_file
  ```

  The chart command is similar to the hchart command. It can take four arguments:
  1. SQL command
  2. Input file
  3. Kind of plot from a variety including pie, bar, boxplot, etc.
  4. Output file.

  Example:

  ```python
  %chart -k pie -c "SELECT pclass, AVG(age) AS av_avg FROM titanic GROUP BY 1;"
  ```

- Many new metrics for classification and regression

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

- Import structures have changed

  Another major change is that the code has been completely restructured. This means that, going forward, all the imports will now be done differently. Note that currently, we still allow the previous structure of import as well, but it will gradually get deprecated.

  According to the new structure, we have the following parent folders:

  - Core [includes `vDataFrame`, parsers `string_sql`, and `tablesample`]
  - Machine Learning [includes model selection, metrics, memmodels, and also all the ML functions of Vertica]
  - SQL [includes dtypes, insert, drop, etc.]
  - Jupyter [includes extensions such as magic SQL and magic chart]
  - Datasets [includes loaders and sample datasets]
  - Connection [includes connect, read, write, etc.]
  - _config [includes configurations]
  - _utils [icnludes all utilities]

  So now, in order to use Vertica's `LinearRegression`, it should be imported as follows:

  ```python
  from verticapy.machine_learning.vertica import LinearRegression
  ```

  Another example for importing statistical tests:

  ```python
  from verticapy.machine_learning.model_selection.statistical_tests import het_arch
  ```

Other changes that do not affect the user experience include:

- Restructuring the code to improve readability and better collaboration using PEP8 standard
- Improved the code pylint score to 9+ which makes the code more professional and efficient
- Improved thorough Unit Tests that require considerably less time to compute, making the CI/CD pipeline more efficient.


- Verticapylab autoconnection

Slighlt modification to allow smooth integration of upcoming VerticaPyLab.

## Internal
- A better method to collect usage statistics

A python decorator (@save_verticapy_logs) is used to effectively log the usage statistics of all the functions.

- Plotting class

A set of common classes are created for effective collaboration and incorporation of other plotting libraries int he future

- 



#####################

######################
- Other configurations

Apart from plotting libraries, users can now change the default configurations very conveniently using the config module. Some of the options that can be change include: number of maximum columns, maximum rows, turning sql on etc. 

For example, below the temperory schema is change from "public" to "private".

```python
from verticapy import set_option
set_option("temp_schema", "private")
```

The current state can also be easily fetched using get_option:

```python
from verticapy import set_option
get_option("temp_schema")

```

# Change log

This release includes some major changes including:

	• Machine Learning Support
        -KPrototypes are now supported in VerticaPy

	• vDataFramesSQL is deprecated. Now vDataFrame can be used directly to create vdataframes from SQL

    New format:

```python
    import verticapy as vp
    vp.vDataFrame("(SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2) x")
```

    The new format also allows other ways to create a vDataFrame. 

```python
    vp.vDataFrame({"X":[1,2,3],"Y":['a','b','c']})
```

	• Plotly integration

    The new release comes with better visualziation using plotly as the deault plotting library. Now the plots are more interactive to enhance user experience.

	• Convenient switching of plotting libraries

    The plotting libraries can now easily be switched using the following syntax:

```
import verticapy._config.config as conf
conf.set_option("plotting_lib","matplotlib")
```

    Note that the function "Hchart" is also deprecated as the highcharts plots can be plotted using the regular plotting syntax.

	• Direct plotting from SQL queries

    In this update, users can directly plot from SQL queries.

```python
%load_ext verticapy.jupyter.extensions.chart_magic
%chart -c sql_command -f input_file -k 'auto' -o output_file
```

    The chart command is similar to the hchart command. It can take four arguments (1) SQL command, (2) Input file, (3) Kind of plot from a variety including pie, bar, boxplot etc, (4) Output file. 

    Example:
```
%chart -k pie -c "SELECT pclass, AVG(age) AS av_avg FROM titanic GROUP BY 1;"
```

	• Many new metrics for classification and regression

    New metrics added to classificaiton reprot:
        - Akaike’s  Information  Criterion (aic)
        - Balanced Accuracy (ba)
        - False Discovery Rate (fdr)
        - Fowlkes–Mallows index
        - Positive Likelihood Ratio 
        - Negative Likelihood Ratio
        - Prevalence Threshold
        - Specificity

    Most of the above metrics are new in this version and can be accessed directly.

    New metrics added to regression report:
        - Mean Squared Log Error
        - Quantile Error

	• Import structures

    With this new release, we have completely restructure the code so that means that all the imports will now be done differently. Note that currently, we allow previous structure of import as well, but it will gradually get deprecated. 

    According to the new structure we have the following parent folders

    Core [includes vDataFrame, parsers string_sql and tablesample]
    Machine Learning [includes model selection, metrics, memmodels, and also all the ML functions of Vertica]
    SQL [includes dtypes, insert, drop etc.]
    Jupyter [includes extensions such as magic sql and magic chart]
    Datasets [inludes loaders and sample datasets]
    Connection [includes connect, read, write etc.]

    So now, in order to use Vertica's "LinearRegression", it should be imported as follows:

```python
from verticapy.machine_learning.vertica import LinearRegression
```

    Another example for import statistitcal tests:

```python
from verticapy.machine_learning.model_selection.statistical_tests import het_arch
```

Other changes that do not affect the user experience include:

* Restructuring the code to improve readability and better collaboration using PEP8 standard
* Improved the code pylint score to 9+ which makes the code more professional and efficient
* Improved thorough Unit Tests that require considerably less time to compute making the CI/CD pipelien more efficient    
	