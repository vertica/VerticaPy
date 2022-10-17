Thank you for considering contributing to *VerticaPy* and helping to make it even better than it is today!

This document will guide you through the contribution process. There are a number of ways you can help:

 - [Bug Reports](#bug-reports)
 - [Feature Requests](#feature-requests)
 - [Code Contributions](#code-contributions)
 
# Bug Reports

If you find a bug, submit an [issue](https://github.com/vertica/VerticaPy/issues) with a complete and reproducible bug report. If the issue can't be reproduced, it will be closed. If you opened an issue, but figured out the answer later on your own, comment on the issue to let people know, then close the issue.

For issues (e.g. security related issues) that are **not suitable** to be reported publicly on the GitHub issue system, report your issues to [Vertica open source team](mailto:vertica-opensrc@microfocus.com) directly or file a case with Vertica support if you have a support account.

# Feature Requests

Feel free to share your ideas for how to improve *VerticaPy*. We’re always open to suggestions.
You can open an [issue](https://github.com/vertica/VerticaPy/issues)
with details describing what feature(s) you'd like to be added or changed.

If you would like to implement the feature yourself, open an issue to ask before working on it. Once approved, please refer to the [Code Contributions](#code-contributions) section.

# Code Contributions

## Step 1: Fork

Fork the project [on Github](https://github.com/vertica/VerticaPy) and check out your copy locally.

```shell
git clone git@github.com:YOURUSERNAME/VerticaPy.git
cd VerticaPy
```

Your GitHub repository **YOURUSERNAME/VerticaPy** will be called "origin" in
Git. You should also setup **vertica/VerticaPy** as an "upstream" remote.

```shell
git remote add upstream git@github.com:vertica/VerticaPy.git
git fetch upstream
```

### Configure Git for the first time

Make sure git knows your [name](https://help.github.com/articles/setting-your-username-in-git/ "Set commit username in Git") and [email address](https://help.github.com/articles/setting-your-commit-email-address-in-git/ "Set commit email address in Git"):

```shell
git config --global user.name "John Smith"
git config --global user.email "email@example.com"
```

## Step 2: Branch

Create a new branch for the work with a descriptive name:

```shell
git checkout -b my-fix-branch
```

## Step 3: Install dependencies

Install the Python dependencies for development:

```shell
pip3 install -r requirements-dev.txt
```

## Step 4: Get the test suite running (Under development)

*VerticaPy* comes with its own test suite in the `verticapy/tests` directory. It’s our policy to make sure all tests pass at all times.

We appreciate any and all [contributions to the test suite](#tests)! These tests use a Python module: [pytest](https://docs.pytest.org/en/latest/). You might want to check out the pytest documentation for more details.

You must have access to a Vertica database to run the tests. We recommend using a non-production database, because some tests may need the superuser permission to manipulate global settings and potentially break that database. Heres one way to go about it:
- Download docker kitematic: https://kitematic.com/
- Spin up a vertica container (e.g. sumitchawla/vertica)

Spin up your Vertica database for tests and then config test settings:
* Here are default settings:
  ```sh
  host: 'localhost'
  port: 5433
  user: <current OS login user>
  database: <same as the value of user>
  password: ''
  log_dir: 'vp_test_log'  # all test logs would write to files under this directory
  log_level: logging.WARNING
  ```
* Override with a configuration file called `verticapy/tests/common/vp_test.conf`. This is a file that would be ignored by git. We created an example `verticapy/tests/common/vp_test.conf.example` for your reference.
  ```sh
  # edit under [vp_test_config] section
  VP_TEST_HOST=10.0.0.2
  VP_TEST_PORT=5000
  VP_TEST_USER=dbadmin
  VP_TEST_DATABASE=vdb1
  VP_TEST_PASSWORD=abcdef1234
  VP_TEST_LOG_DIR=my_log/year/month/date
  VP_TEST_LOG_LEVEL=DEBUG
  ```
* Override again with VP_TEST_* environment variables
  ```shell
  # Set environment variables in linux
  $ export VP_TEST_HOST=10.0.0.2
  $ export VP_TEST_PORT=5000
  $ export VP_TEST_USER=dbadmin
  $ export VP_TEST_DATABASE=vdb1
  $ export VP_TEST_PASSWORD=abcdef1234
  $ export VP_TEST_LOG_DIR=my_log/year/month/date
  $ export VP_TEST_LOG_LEVEL=DEBUG

  # Delete your environment variables after tests
  $ unset VP_TEST_PASSWORD
  ```

Tox (https://tox.readthedocs.io) is a tool for running those tests in different Python environments. *VerticaPy*
includes a `tox.ini` file that lists all Python versions we test. Tox is installed with the `requirements-dev.txt`,
discussed above.

Edit `tox.ini` envlist property to list the version(s) of Python you have installed. Then you can run the **tox** command from any place in the *verticapy* source tree. If VP_TEST_LOG_DIR sets to a relative path, it will be in the *verticapy* directory no matter where you run the **tox** command.

Examples of running tests:

```bash
# Run all tests using tox:
tox

# Run tests on specified python versions with `tox -e ENV,ENV`
tox -e py36,py37

# Run specific tests by filename (e.g.) `test_vDF_combine_join_sort.py`
tox -- verticapy/tests/vDataFrame/test_vDF_combine_join_sort.py

# Run all tests on the python version 3.6:
tox -e py36 -- verticapy/tests

# Run all tests on the python version 3.7 with verbose result outputs:
tox -e py37 -v -- verticapy/tests

# Run an individual test on specified python versions.
# e.g.: Run the test `test_vDF_append` under `test_vDF_combine_join_sort.py` on the python versions 3.7 and 3.8
tox -e py37,py38 -- verticapy/tests/vDataFrame/test_vDF_combine_join_sort.py::TestvDFCombineJoinSort::test_vDF_append
```

The arguments after the `--` will be substituted everywhere where you specify `{posargs}` in your test *commands* of
`tox.ini`, which are sent to pytest. See `pytest --help` to see all arguments you can specify after the `--`.

You might also run `pytest` directly, which will evaluate tests in your current Python environment, rather than across
the Python environments/versions that are enumerated in `tox.ini`.

For more usages about [tox](https://tox.readthedocs.io), see the Python documentation.

## Step 5: Implement your fix or feature

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first.

### Useful Functions

This section is an overview of some useful functions. You can use these to implement new features.

To check if a list of columns belongs to the vDataFrame:
```python
# import
from verticapy import vDataFrame

# Function
vDataFrame.are_namecols_in(columns: Union[str, list]):
    """
---------------------------------------------------------------------------
Method used to check if the input column names are used by the vDataFrame.
If not, the function raises an error.

Parameters
----------
columns: list/str
    List of columns names.
    """

# Example
# if vDataFrame 'vdf' has two columns named respectively 'A' and 'B'
# vDataFrame.are_namecols_in(['A', 'B']) will work.
# vDataFrame.are_namecols_in(['A', 'C']) will raise an error.
```

To format a list using the columns of the vDataFrame:
```python
# import
from verticapy import vDataFrame

# Function
vDataFrame.format_colnames(self, columns: Union[str, list]):
    """
---------------------------------------------------------------------------
Method used to format a list of columns with the column names of the 
vDataFrame.

Parameters
----------
columns: list/str
    List of column names to format.

Returns
-------
list
    Formatted column names.
    """

# Example
# if vDataFrame 'vdf' has two columns named respectively 'CoLuMn A' and 'CoLumnB'
# vDataFrame.format_colnames(['column a', 'columnb']) == ['CoLuMn A', 'CoLumnB']
```

Identifiers in a SQL query must be formatted a certain way. You can use the following function to get a properly formatted identifier:

```python
# import 
from verticapy import quote_ident

# Function
def quote_ident(column: str):
  """
---------------------------------------------------------------------------
Returns the specified string argument in the required format to use it as 
an identifier in a SQL query.

Parameters
----------
column: str
    Column name.

Returns
-------
str
    Formatted column name.
  """

# Example
# quote_ident('my column name') == '"my column name"'
```

The two following functions will generate the VerticaPy logo as a string or as an HTML object.
```python
# import
from verticapy.logo import gen_verticapy_logo_html
from verticapy.logo import gen_verticapy_logo_str

# Functions
def gen_verticapy_logo_html() # VerticaPy HTML Logo
def gen_verticapy_logo_str()  # VerticaPy Python STR Logo
```

### Feature Example

The vDataFrame a powerful Python object that lies at the heart of VerticaPy. vDataFrames consist of vColumn objects that represent columns in the dataset.

You can find the vDataFrame's many methods in vdataframe.py under the '# Methods' comment.

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/vdf_file.png' width="60%">
</p>

You can define any new vDataFrame method after this comment. The same applies to vColumns.

For any method definition, you should note the type hints for every variable. For variables of multiple types, use the union operator.

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/function.png' width="60%">
</p>

Be sure to write a detailed description for each function that explains how it works.

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/description.png' width="60%">
</p>

Uses the check_types() function to verify the types of each parameter, vDataFrame.are_namecols_in() to verify that the specified column belongs to the main vDataFrame, and vDataFrame.format_colnames() to format column names.

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/check_types.png' width="60%">
</p>

When using check_types(), create tuples with the variable name, the variable, and a list of the different types:

```python
from verticapy import *

x = 1

# Correct Type
check_types([("x", x, [int])])
```

```python
# Incorrect Type
check_types([("x", x, [str])])
```
```
/Users/Badr/Library/Python/3.6/lib/python/site-packages/verticapy/toolbox.py:252: Warning: Parameter 'x' must be of type <class 'str'>, found type <class 'int'>
  warnings.warn(warning_message, Warning)
```

To add a parameter with specific values, use a list:

```python
x = "apple"

# Correct parameter
check_types([("x", x, ["apple", "banana", "lemon"])])
```

```python
x = "apple"

# Incorrect parameter
check_types([("x", x, ["potato", "tomato", "salad"])])
```
```
/Users/Badr/Library/Python/3.6/lib/python/site-packages/verticapy/toolbox.py:236: Warning: Parameter 'x' must be in [potato|tomato|salad], found 'apple'
  warnings.warn(warning_message, Warning)
```

Remember: the vDataFrame.are_namecols_in() and vDataFrame.format_colnames() functions are essential for correctly formated input column names.

```python
# Displaying columns from the titanic dataset
from verticapy.datasets import load_titanic
titanic = load_titanic()
titanic.get_columns()
```
```
['"pclass"',
 '"survived"',
 '"name"',
 '"sex"',
 '"age"',
 '"sibsp"',
 '"parch"',
 '"ticket"',
 '"fare"',
 '"cabin"',
 '"embarked"',
 '"boat"',
 '"body"',
 '"home.dest"']
```
```python
# Verify that the 'boat' column is inside the vDataFrame
titanic.are_namecols_in(["boat"])
```
```python
# Verify that the 'wrong_name' column is not inside the vDataFrame
titanic.are_namecols_in(["wrong_name"])
```
```
---------------------------------------------------------------------------
MissingColumn                             Traceback (most recent call last)
<ipython-input-5-532ff8a7a7a7> in <module>()
----> 1 titanic.are_namecols_in(["wrong_name"])

/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/verticapy/vdataframe.py in are_namecols_in(self, columns)
   1456                 raise MissingColumn(
   1457                     "The Virtual Column '{}' doesn't exist{}.".format(
-> 1458                         column.lower().replace('"', ""), e
   1459                     )
   1460                 )

MissingColumn: The Virtual Column 'wrong_name' doesn't exist.
```
```python
# vDataFrame.format_colnames() automatically formats names
titanic.format_colnames(["BoAt"])
```
```
['"boat"']
```
Use the \_genSQL\_ method to get the current vDataFrame relation.
```python
titanic.__genSQL__()
```
```
'"public"."titanic"'
```
And the \_executeSQL\_ function to execute a SQL query.
```python
from verticapy import executeSQL
executeSQL("SELECT * FROM {0} LIMIT 2".format(titanic.__genSQL__()))
```
```
<vertica_python.vertica.cursor.Cursor at 0x115f972e8>
```
The result of the query is accessible using one of the methods of the 'executeSQL' parameter.
```python
executeSQL("SELECT * FROM {0} LIMIT 2".format(titanic.__genSQL__()), method="fetchall")
```
```
[[1,
  0,
  'Allison, Miss. Helen Loraine',
  'female',
  Decimal('2.000'),
  1,
  2,
  '113781',
  Decimal('151.55000'),
  'C22 C26',
  'S',
  None,
  None,
  'Montreal, PQ / Chesterville, ON'],
 [1,
  0,
  'Allison, Mr. Hudson Joshua Creighton',
  'male',
  Decimal('30.000'),
  1,
  2,
  '113781',
  Decimal('151.55000'),
  'C22 C26',
  'S',
  None,
  135,
  'Montreal, PQ / Chesterville, ON']]
```
The last thing to know before writing an entire function is the save_to_query_profile() method. It is used to save the method call in a specific Vertica table. This table can be used to run statistics or in order to debug some code.

The function is quite simple and it uses 3 main parameters:
 - name: it is the name of the function.
 - path: it is the location of the function. For example, the method corr of the vDataFrame is located in vdataframe.vDataFrame.
 - json_dict: it is the dictionary of the different parameters.

You'll understand quickly how to use the function in the following examples.

For example, let's create a method to compute the correlations between two vDataFrame columns.
```python
# Example correlation method for a vDataFrame

# Add type hints
def pearson(self, column1: str, column2: str):
    # Describe the function
    """
    ---------------------------------------------------------------------------
    Computes the Pearson Correlation Coefficient of the two input vColumns. 

    Parameters
    ----------
    column1: str
        Input vColumn.
    column2: str
        Input vColumn.

    Returns
    -------
    Float
        Pearson Correlation Coefficient

    See Also
    --------
    vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
        """
    # Save the call in the query profile table
    save_to_query_profile(name="pearson", # name of the function - pearson
                          path="vdataframe.vDataFrame", # function location
                          json_dict={"column1": column1, # function parameters
                                     "column2": column2,},)
    # Check data types
    check_types([("column1", column1, [str]),
                 ("column2", column2, [str]),])
    # Check if the columns belong to the vDataFrame
    self.are_namecols_in([column1, column2])
    # Format the columns
    column1, column2 = self.format_colnames([column1, column2])
    # Get the current vDataFrame relation
    table = self.__genSQL__()
    # Create the SQL statement - Do not forget to label the query whenever it is possible
    query = f"SELECT /*+LABEL(vDataFrame.pearson)*/ CORR({column1}, {column2}) FROM {table};"
    # Execute the SQL query and get the result
    result = executeSQL(query, 
                        title = "Computing Pearson coefficient", 
                        method="fetchfirstelem")
    # Return the result
    return result
```
Same can be done with vColumn methods.
```python
# Example Method for a vColumn

# Add types hints
def pearson(self, column: str,):
    # Describe the function
    """
    ---------------------------------------------------------------------------
    Computes the Pearson Correlation Coefficient of the vColumn and the input 
    vColumn. 

    Parameters
    ----------
    column: str
        Input vColumn.

    Returns
    -------
    Float
        Pearson Correlation Coefficient

    See Also
    --------
    vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
        """
    # Save the call in the query profile table
    save_to_query_profile(name="pearson", 
                          path="vcolumn.vColumn", 
                          json_dict={"column": column,},)
    # Check data types
    check_types([("column", column, [str]),])
    # Check if the column belongs to the vDataFrame 
    # self.parent represents the vColumn parent
    self.parent.are_namecols_in([column])
    # Format the column
    column1 = self.parent.format_colnames([column])[0]
    # Get the current vColumn name
    column2 = self.alias
    # Get the current vDataFrame relation
    table = self.parent.__genSQL__()
    # Create the SQL statement - Do not forget to label the query whenever it is possible
    query = f"SELECT /*+LABEL(vColumn.pearson)*/ CORR({column1}, {column2}) FROM {table};"
    # Execute the SQL query and get the result
    result = executeSQL(query, 
                        title = "Computing Pearson coefficient", 
                        method="fetchfirstelem")
    # Return the result
    return result
```
Functions will work exactly the same.
```python
# Example function

# Add type hints
def pearson(vdf: vDataFrame, column1: str, column2: str):
    # Describe the function
    """
    ---------------------------------------------------------------------------
    Computes the Pearson Correlation Coefficient of the two input vColumns. 

    Parameters
    ----------
    vdf: vDataFrame
        Input vDataFrame.
    column1: str
        Input vColumn.
    column2: str
        Input vColumn.

    Returns
    -------
    Float
        Pearson Correlation Coefficient

    See Also
    --------
    vDataFrame.corr : Computes the Correlation Matrix of the vDataFrame.
        """
    # Save the call in the query profile table
    save_to_query_profile(name="pearson", 
                          path="vdataframe", 
                          json_dict={"vdf": vdf,
                                     "column1": column1,
                                     "column2": column2,},)
    # Check data types
    check_types([("vdf", vdf, [vDataFrame]),
                 ("column1", column1, [str]),
                 ("column2", column2, [str]),])
    # Check if the columns belong to the vDataFrame
    vdf.are_namecols_in([column1, column2])
    # Format the columns
    column1, column2 = vdf.format_colnames([column1, column2])
    # Get the current vDataFrame relation
    table = vdf.__genSQL__()
    # Create the SQL statement - Do not forget to label the query whenever it is possible
    query = f"SELECT /*+LABEL(pearson)*/ CORR({column1}, {column2}) FROM {table};"
    # Execute the SQL query and get the result
    result = executeSQL(query, 
                        title = "Computing Pearson coefficient", 
                        method="fetchfirstelem")
    # Return the result
    return result
```

Functions must include unit tests. Unit tests are located in the 'tests' folder and sorted by theme. Unit tests can be tested with the default VerticaPy datasets.

For the function we just created, we would place the unit tests 'test_vDF_correlation.py' in the 'vDataFrame' directory.

A unit test might look like this:
```python
# Example unit test function

def test_pearson(self, titanic_vd):
  result_1 = titanic_vd.pearson("age", "fare")
  result_2 = titanic_vd.pearson("age", "survived")
  # we can test approximately the result of our function for some known values
  assert result == pytest.approx(0.178575164117464, 1e-2)
  assert result == pytest.approx(-0.0422446185581737, 1e-2)
```

You are now ready to create your first contribution!

### License Headers

Every file in this project must use the following Apache 2.0 header (with the appropriate year or years in the "[yyyy]" box; if a copyright statement from another party is already present in the code, you may add the statement on top of the existing copyright statement):

```
Copyright (c) [yyyy] Micro Focus or one of its affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Commits

Make some changes on your branch, then stage and commit as often as necessary:

```shell
git add .
git commit -m 'Added two more tests for #166'
```

When writing the commit message, try to describe precisely what the commit does. The commit message should be in lines of 72 chars maximum. Include the issue number `#N`, if the commit is related to an issue.

### Tests

Add appropriate tests for the bug’s or feature's behavior, run the test suite again and ensure that all tests pass. Here is the guideline for writing test:
 - Tests should be easy for any contributor to run. Contributors may not get complete access to their Vertica database, for example, they may only have a non-admin user with write privileges to a single schema, and the database may not be the latest version. We encourage tests to use only what they need and nothing more.
 - If there are requirements to the database for running a test, the test should adapt to different situations and never report a failure. For example, if a test depends on a multi-node database, it should check the number of DB nodes first, and skip itself when it connects to a single-node database (see helper function `require_DB_nodes_at_least()` in `verticapy/tests/integration_tests/base.py`).

## Step 6: Push and Rebase

You can publish your work on GitHub just by doing:

```shell
git push origin my-fix-branch
```

When you go to your GitHub page, you will notice commits made on your local branch is pushed to the remote repository.

When upstream (vertica/VerticaPy) has changed, you should rebase your work. The **rebase** command creates a linear history by moving your local commits onto the tip of the upstream commits.

You can rebase your branch locally and force-push to your GitHub repository by doing:

```shell
git checkout my-fix-branch
git fetch upstream
git rebase upstream/master
git push -f origin my-fix-branch
```


## Step 7: Make a Pull Request

When you think your work is ready to be pulled into *VerticaPy*, you should create a pull request(PR) at GitHub.

A good pull request means:
 - commits with one logical change in each
 - well-formed messages for each commit
 - documentation and tests, if needed

Go to https://github.com/YOURUSERNAME/VerticaPy and [make a Pull Request](https://help.github.com/articles/creating-a-pull-request/) to `vertica:master`. 

### Sign the CLA
Before we can accept a pull request, we first ask people to sign a Contributor License Agreement (or CLA). We ask this so that we know that contributors have the right to donate the code. You should notice a comment from **CLAassistant** on your pull request page, follow this comment to sign the CLA electronically. 

### Review
Pull requests are usually reviewed within a few days. If there are comments to address, apply your changes in new commits, rebase your branch and force-push to the same branch, re-run the test suite to ensure tests are still passing. We care about quality, Vertica has internal test suites to run as well, so your pull request won't be merged until all internal tests pass. In order to produce a clean commit history, our maintainers would do squash merging once your PR is approved, which means combining all commits of your PR into a single commit in the master branch.

That's it! Thank you for your code contribution!

After your pull request is merged, you can safely delete your branch and pull the changes from the upstream repository.



