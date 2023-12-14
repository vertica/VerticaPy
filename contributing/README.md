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
tox -e py39,py310

# Run specific tests by filename (e.g.) `test_join_union_sort.py`
tox -- verticapy/tests_new/core/vdataframe/test_join_union_sort.py

# Run all tests on the python version 3.9:
tox -e py39 -- verticapy/tests

# Run all tests on the python version 3.10 with verbose result outputs:
tox -e py310 -v -- verticapy/tests

# Run an individual test on specified python versions.
# e.g.: Run the test `test_append` under `test_join_union_sort.py` on the python versions 3.9 and 3.10
tox -e py39,py310 -- verticapy/tests_new/core/vdataframe/test_join_union_sort.py::TestJoinUnionSort::test_append
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
vDataFrame.get_columns():
    """
    Returns the TableSample columns.

    Returns
    -------
    list
        columns.
    """

# Example
# if vDataFrame 'vdf' has two columns named respectively 'A' and 'B'
# vDataFrame.get_columns()) will return a list: ["A","B"].
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
from verticapy._utils._logo import verticapy_logo_html 
from verticapy._utils._logo import verticapy_logo_str

# Functions
def verticapy_logo_html() # VerticaPy HTML Logo
def verticapy_logo_str()  # VerticaPy Python STR Logo
```

### Feature Example

The vDataFrame a powerful Python object that lies at the heart of VerticaPy. vDataFrames consist of vColumn objects that represent columns in the dataset.

You can find all vDataFrame's methods inside the folder verticapy/core/vdataframe. Note that similar methods have been clubbed together inside one module/file. For examples, all methods pertaining to aggregates are in the '_aggregate.py' file.


You can define any new vDataFrame method inside these modules depending on the nature of the method. The same applies to vColumns. You can use any of the developed classes to inherit properties.

When defining a function, you should specify the 'type' hints for every variable: 

- For variables of multiple types, use the Union operator.
- For variables that are optional, use the Optional operator. 
- For variables that require literal input, use the Literal operator. 

There are examples of such hints throughout the code. 


<p align="center">
<img src='https://github.com/vertica/VerticaPy/assets/46414488/149e4265-62e9-4d09-a58f-1f62fcd354cc' width="60%">
</p>

Be sure to write a detailed description for each function that explains how it works.

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/description.png' width="60%">
</p>


Important: the vDataFrame.get_columns() and vDataFrame.format_colnames() functions are essential for correctly formatting input column names.

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



Use the \_genSQL method to get the current vDataFrame relation.
```python
titanic._genSQL()
```
```
'"public"."titanic"'
```
And the \_executeSQL\_ function to execute a SQL query.
```python
from verticapy._utils._sql._sys import _executeSQL
_executeSQL(f"SELECT * FROM {titanic._genSQL()} LIMIT 2")
```
```
<vertica_python.vertica.cursor.Cursor at 0x115f972e8>
```
The result of the query is accessible using one of the methods of the 'executeSQL' parameter.
```python
_executeSQL(f"SELECT * FROM {titanic._genSQL()} LIMIT 2",method="fetchall")
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
The @save_verticapy_logs decorator saves information about a specified VerticaPy method to the QUERY_PROFILES table in the Vertica database. You can use this to collect usage statistics on methods and their parameters.

For example, to create a method to compute the correlations between two vDataFrame columns:
```python
# Example correlation method for a vDataFrame

# Add type hints + @save_verticapy_logs decorator
@save_verticapy_logs
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
    # Check data types
    # Format the columns
    column1, column2 = self.format_colnames([column1, column2])
    # Get the current vDataFrame relation
    table = self._genSQL()
    # Create the SQL statement - Label the query when possible
    query = f"SELECT /*+LABEL(vDataFrame.pearson)*/ CORR({column1}, {column2}) FROM {table};"
    # Execute the SQL query and get the result
    result = _executeSQL(query, 
                        title = "Computing Pearson coefficient", 
                        method="fetchfirstelem")
    # Return the result
    return result
```
Same can be done with vColumn methods.
```python
# Example Method for a vColumn

# Add types hints + @save_verticapy_logs decorator
@save_verticapy_logs
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
    # Format the column
    column1 = self.parent.format_colnames([column])[0]
    # Get the current vColumn name
    column2 = self.alias
    # Get the current vDataFrame relation
    table = self.parent._genSQL()
    # Create the SQL statement - Label the query when possible
    query = f"SELECT /*+LABEL(vColumn.pearson)*/ CORR({column1}, {column2}) FROM {table};"
    # Execute the SQL query and get the result
    result = executeSQL(query, 
                        title = "Computing Pearson coefficient", 
                        method="fetchfirstelem")
    # Return the result
    return result
```
Functions will work exactly the same.

### Unit Tests

Functions must include unit tests, which are located in the 'tests' folder. The test files follow the same folder structure as the original veritcapy directory. For each file there is a test file. That means if a function is added in core/vdataframe/_aggregate.py file then its respective test will be added in the test file test/core/vdataframe/test_aggregate.py.

Unit tests can be tested with the default VerticaPy datasets or the smaller efficient datasets creted inside the tests/conftest.py file. Be sure to look at all the datasets before creating your own. All these datasets can be imported as fixtures:

```python
def test_properties_type(self, titanic_vd):
    result=titanic_vd["survived"].bar()
    assert(type(result)==plotly.graph_objs._figure.Figure)
```
The above titanic_vd is a fixture that is defined in the conftest file. 

To make the code efficient, it is highly encouraged to use fixtures and to share results across the scope. This ensures that the result will be shared. For every test (i.e. assert), there should be a separate function. To combine multiple tests for the same method, feel free to use classes. 

For the function we just created, we would place the unit tests 'test_vDF_correlation.py' in the 'vDataFrame' directory.

A unit test might look like this:
```python
# Example unit test function

class TestPearson(self):
    def test_age_and_fare(titanic_vd):
        result= titanic_vd.pearson("age", "fare")
        assert result == pytest.approx(0.178575164117464, 1e-2)

    def test_age_and_survived(titanic_vd):
        result_2 = titanic_vd.pearson("age", "survived")
        assert result == pytest.approx(-0.0422446185581737, 1e-2)  
```

A fixture by the name of "schema_loader" has been defined in the conftest file that creates a schema with a random name. This schema is dropped at the end of the unit test. You are encouraged to make use of this fixture to name your models/datasets, if necessary. 
For example, the follwoing loads a dataset and gives it a name from the schema.

```python
@pytest.fixture(scope="module")
def titanic_vd(schema_loader):
    """
    Create a dummy vDataFrame for titanic dataset
    """
    titanic = load_titanic(schema_loader, "titanic")
    yield titanic
    drop(name=f"{schema_loader}.titanic")
```
Since we are using the "schema_loader" fixture, we do not necessarily have to drop the dataset schema because it is automatically dropped at the end of the unit test.

Lastly, double check to make sure that your test allows parallel execution by using the following pytest command:

```python
pytest -n auto --dist=loadscope
```
Note that in order to use the above, you will have to install pytest-xdist.


Add appropriate tests for the bugs or features behavior, run the test suite again, and ensure that all tests pass. Here are additional guidelines for writing tests:
 - Tests should be easy for any contributor to run. Contributors may not get complete access to their Vertica database. For example, they may only have a non-admin user with write privileges to a single schema, and the database may not be the latest version. We encourage tests to use only what they need and nothing more.
 - If there are requirements to the database for running a test, the test should adapt to different situations and never report a failure. For example, if a test depends on a multi-node database, it should check the number of DB nodes first, and skip itself when it connects to a single-node database (see helper function `require_DB_nodes_at_least()` in `verticapy/tests/integration_tests/base.py`).

### Code formatting as per PEP 8

Once you are satisfied with your code, please run [black](https://black.readthedocs.io/en/stable/) for your code. Black will automatically format all your code to make it professional and consistent with PEP 8.

Next please run [pylint](https://pypi.org/project/pylint/) and ensure that your score is above the minimum threshold of 5. Pylint will automatically provide you with the improvement opportunities that you can adjust to increaes your score.

As per the updated CI/CD, no code will be accepted that requires formatting using black or has a lower pylint score than the threshold stated above. 

### License Headers

Every file in this project must use the following Apache 2.0 header (with the appropriate year or years in the "[yyyy]" box; if a copyright statement from another party is already present in the code, you may add the statement on top of the existing copyright statement):

```
Copyright  (c)  2018-2023 Open Text  or  one  of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
```
You are now ready to make your first contribution!

### Commits

Make some changes on your branch, then stage and commit as often as necessary:

```shell
git add .
git commit -m 'Added two more tests for #166'
```

When writing the commit message, try to describe precisely what the commit does. The commit message should be in lines of 72 chars maximum. Include the issue number `#N`, if the commit is related to an issue.


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
 - a self-explanatory title (and the content of the PR should not go beyond the original title/scope)
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



