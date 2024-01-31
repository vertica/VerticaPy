<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/assets/img/logo.png' width="180px">
</p>

:star: 2023-12-01: VerticaPy secures 200 stars.

:loudspeaker: 2020-06-27: Vertica-ML-Python has been renamed to VerticaPy.

:warning: The old website has been relocated to https://www.vertica.com/python/old/ and will be discontinued soon. We strongly recommend upgrading to the new major release before August 2024.

:warning: The following README is for VerticaPy 1.0.x and onwards, and so some of the elements may not be present in the previous versions.

:scroll: Some basic syntax can be found in [the cheat sheet](assets/cheat_sheet/).

📰 Check out the latest newsletter [here](assets/news_letter/1.0.x/VerticaPy_Newsletter.pdf).

# VerticaPy

[![PyPI version](https://badge.fury.io/py/verticapy.svg)](https://badge.fury.io/py/verticapy)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/verticapy?color=yellowgreen)](https://anaconda.org/conda-forge/verticapy)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/vertica/VerticaPy/branch/master/graph/badge.svg?token=a6GiFYI9at)](https://codecov.io/gh/vertica/VerticaPy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/assets/img/benefits.png' width="92%">
</p>

VerticaPy is a Python library with scikit-like functionality used to conduct data science projects on data stored in Vertica, taking advantage of Vertica’s speed and built-in analytics and machine learning features. VerticaPy offers robust support for the entire data science life cycle, uses a 'pipeline' mechanism to sequentialize data transformation operations, and offers beautiful graphical options.
<br><br>

# Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Connecting to the Database](#connecting-to-the-database)
- [Documentation](#documentation)
- [Use-cases](#use-cases)
- [Highlighted Features](#highllighted-features)
  - [Themes - Dark | Light](#themes)
  - [SQL Magic](#sql-magic)
  - [SQL Plots](#sql-plots)
  - [Diverse Database Connections](#multiple-database-connection-using-dblink)
  - [Python and SQL Combo](#python-and-sql-combo)
  - [Charts](#charts)
  - [Complete ML pipeline](#complete-machine-learning-pipeline)
- [Quickstart](#quickstart)
- [Help and Support](#help-an-support)
  - [Contributing](#contributing)
  - [Communication](#communication)

<br>

# Introduction

Vertica was the first real analytic columnar database and is still the fastest in the market. However, SQL alone isn't flexible enough to meet the needs of data scientists.
<br><br>
Python has quickly become the most popular tool in this domain, owing much of its flexibility to its high-level of abstraction and impressively large and ever-growing set of libraries. Its accessibility has led to the development of popular and perfomant APIs, like pandas and scikit-learn, and a dedicated community of data scientists. Unfortunately, Python only works in-memory as a single-node process. This problem has led to the rise of distributed programming languages, but they too, are limited as in-memory processes and, as such, will never be able to process all of your data in this era, and moving data for processing is prohobitively expensive. On top of all of this, data scientists must also find convenient ways to deploy their data and models. The whole process is time consuming.
<br><br>
**VerticaPy aims to solve all of these problems**. The idea is simple: instead of moving data around for processing, VerticaPy brings the logic to the data.
<br><br>
3+ years in the making, we're proud to bring you VerticaPy.
<br><br>
Main Advantages:
<ul>
 <li> Easy Data Exploration.</li>
 <li> Fast Data Preparation.</li>
 <li> In-Database Machine Learning.</li>
 <li> Easy Model Evaluation.</li>
 <li> Easy Model Deployment.</li>
 <li> Flexibility of using either Python or SQL.</li>
</ul>

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/assets/img/architecture.png' width="92%">
</p>

[:arrow_up: Back to TOC](#table-of-contents)
<br>

## Installation

To install <b>VerticaPy</b> with pip:
```shell
# Latest release version
root@ubuntu:~$ pip3 install verticapy[all]

# Latest commit on master branch
root@ubuntu:~$ pip3 install git+https://github.com/vertica/verticapy.git@master
```
To install <b>VerticaPy</b> from source, run the following command from the root directory:
```shell
root@ubuntu:~$ python3 setup.py install
```

A detailed installation guide is available at: <br>

https://www.vertica.com/python/documentation/installation.html

[:arrow_up: Back to TOC](#table-of-contents)
<br>

## Connecting to the Database

VerticaPy is compatible with several clients. For details, see the <a href='https://www.vertica.com/python/documentation/connection.html'>connection page</a>.<br>

[:arrow_up: Back to TOC](#table-of-contents)
<br>

## Documentation

The easiest and most accurate way to find documentation for a particular function is to use the help function:

```python
import verticapy as vp

help(vp.vDataFrame)
```

Official documentation is available at: <br>

https://www.vertica.com/python/documentation/

[:arrow_up: Back to TOC](#table-of-contents)
<br>

## Use-cases

Examples and case-studies: <br>

https://www.vertica.com/python/examples/

[:arrow_up: Back to TOC](#table-of-contents)
<br>

## Highlighted Features

### Themes

VerticaPy, offers users the flexibility to customize their coding experience with two visually appealing themes: **Dark** and **Light**. 

Dark mode, ideal for night-time coding sessions, features a sleek and stylish dark color scheme, providing a comfortable and eye-friendly environment. 

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/8ee0b717-a994-4535-826a-7ca4db3772b5" width="70%">
</p>

On the other hand, Light mode serves as the default theme, offering a clean and bright interface for users who prefer a traditional coding ambiance. 

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/24757bfd-4d0f-4e92-9aca-45476d704a33" width="70%">
</p>

Theme can be easily switched by:

```python
import verticapy as vp

vp.set_option("theme", "dark") # can be switched 'light'.
```

VerticaPy's theme-switching option ensures that users can tailor their experience to their preferences, making data exploration and analysis a more personalized and enjoyable journey.

[:arrow_up: Back to TOC](#table-of-contents)
<br>

### SQL Magic
You can use VerticaPy to execute SQL queries directly from a Jupyter notebook. For details, see <a href='https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.jupyter.extensions.sql_magic.sql_magic.html#verticapy.jupyter.extensions.sql_magic.sql_magic'>SQL Magic</a>:

#### Example

Load the SQL extension.
```python
%load_ext verticapy.sql
```
Execute your SQL queries.
```sql
%%sql
SELECT version();

# Output
# Vertica Analytic Database v11.0.1-0
```
[:arrow_up: Back to TOC](#table-of-contents)
<br>

### SQL Plots

You can create interactive, professional plots directly from SQL.

To create plots, simply provide the type of plot along with the SQL command.

#### Example
```python
%load_ext verticapy.jupyter.extensions.chart_magic
%chart -k pie -c "SELECT pclass, AVG(age) AS av_avg FROM titanic GROUP BY 1;"
```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/7616ca04-87d4-4fd7-8cb9-015f48fe3c19" width="50%">
</p>

[:arrow_up: Back to TOC](#table-of-contents)
<br>

### Multiple Database Connection using DBLINK

In a single platform, multiple databases (e.g. PostgreSQL, Vertica, MySQL, In-memory) can be accessed using SQL and python.

#### Example
```sql
%%sql
/* Fetch TAIL_NUMBER and CITY after Joining the flight_vertica table with airports table in MySQL database. */
SELECT flight_vertica.TAIL_NUMBER, airports.CITY AS Departing_City
FROM flight_vertica
INNER JOIN &&& airports &&&
ON flight_vertica.ORIGIN_AIRPORT = airports.IATA_CODE;
```
In the example above, the 'flight_vertica' table is stored in Vertica, whereas the 'airports' table is stored in MySQL. We can associate special symbols "&&&" to the different databases to fetch the data. The best part is that all the aggregation is pushed to the databases (i.e. it is not done in memory)!

For more details on how to setup DBLINK, please visit the [github repo](https://github.com/vertica/dblink). To learn about using DBLINK in VerticaPy, check out the [documentation page](https://www.vertica.com/python/documentation/1.0.x/html/notebooks/full_stack/dblink_integration/).

[:arrow_up: Back to TOC](#table-of-contents)
<br>

### Python and SQL Combo

VerticaPy has a unique place in the market because it allows users to use Python and SQL in the same environment. 

#### Example
```python
import verticapy as vp

selected_titanic = vp.vDataFrame(
    "(SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2) x"
)
selected_titanic.groupby(columns=["pclass"], expr=["AVG(AVG)"])
```
[:arrow_up: Back to TOC](#table-of-contents)
<br>

### Charts

Verticapy comes integrated with three popular plotting libraries: matplotlib, highcharts, and plotly.

A gallery of VerticaPy-generated charts is available at:<br>

https://www.vertica.com/python/documentation/chart.html

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/ac62df51-5f26-4b67-839b-fbd962fbaaea" width="70%">
</p>

[:arrow_up: Back to TOC](#table-of-contents)
<br>

### Complete Machine Learning Pipeline

- **Data Ingestion**

  VerticaPy allows users to ingest data from a diverse range of sources, such as AVRO, Parquet, CSV, JSON etc. With a simple command "[read_file](https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.read_file.html)", VerticaPy automatically infers the source type and the data type.

  ```python
  import verticapy as vp

  read_file(
      "/home/laliga/2012.json",
      table_name="laliga",
  )
  ```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/cddc5bbc-9f96-469e-92ee-b4a6e0bc7cfb" width="100%">
</p>
Note: Not all columns are displayed in the screenshot above because of width restriction here.

As shown above, it has created a nested structure for the complex data. The actual file structure is below:

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/6ad242fb-2994-45de-8796-d6af61dae00d" width="30%">
</p>

We can even see the SQL underneath every VerticaPy command by turning on the genSQL option:

```python
  import verticapy as vp

  read_file("/home/laliga/2012.json", table_name="laliga", genSQL=True)
```
```sql
 CREATE LOCAL TEMPORARY TABLE "laliga"
    ("away_score" INT, 
     "away_team" ROW("away_team_gender" VARCHAR, 
                     "away_team_group"  VARCHAR, 
                     "away_team_id"     INT, ... 
                                        ROW("id"   INT, 
                                            "name" VARCHAR)), 
     "competition" ROW("competition_id"   INT, 
                       "competition_name" VARCHAR, 
                       "country_name"     VARCHAR), 
     "competition_stage" ROW("id"   INT, 
                             "name" VARCHAR), 
     "home_score" INT, 
     "home_team" ROW("country" ROW("id"   INT, 
                                   "name" VARCHAR), 
                     "home_team_gender" VARCHAR, 
                     "home_team_group"  VARCHAR, 
                     "home_team_id"     INT, ...), 
     "kick_off"     TIME, 
     "last_updated" DATE, 
     "match_DATE"   DATE, 
     "match_id"     INT, ... 
                    ROW("data_version"          DATE, 
                        "shot_fidelity_version" INT, 
                        "xy_fidelity_version"   INT), 
     "season" ROW("season_id"   INT, 
                  "season_name" VARCHAR)) 
     ON COMMIT PRESERVE ROWS
     COPY "v_temp_schema"."laliga" 
     FROM '/home/laliga/2012.json' 
     PARSER FJsonParser()
```

VerticaPy provides functions for importing other specific file types, such as [read_json](#https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.read_json.html#verticapy.read_json) and [read_csv](#https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.read_csv.html). Since these functions focus on a particular file type, they offer more options for tackling the data. For example, [read_json](#https://www.vertica.com/python/documentation/1.0.x/html/api/verticapy.read_json.html#verticapy.read_json) has a "flatten_arrays" parameter that allows you to flatten nested JSON arrays.

- **Data Exploration**

  There are many options for descriptive and visual exploration. 

```python
from verticapy.datasets import load_iris

iris_data = load_iris()
iris_data.scatter(
    ["SepalWidthCm", "SepalLengthCm", "PetalLengthCm"], 
    by="Species", 
    max_nb_points=30
)
```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/b70bfbf4-22fa-40f9-9958-7fd19dbfc61b" width="40%">
</p>

The <b>Correlation Matrix</b> is also very fast and convenient to compute. Users can choose from a wide variety of correaltions, including cramer, spearman, pearson etc.

```python
from verticapy.datasets import load_titanic

titanic = load_titanic()
titanic.corr(method="spearman")
```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/fd34aac7-890a-484e-a3bc-9173bffa79d2" width="75%">
</p>

By turning on the SQL print option, users can see and copy SQL queries:

```python
from verticapy import set_option

set_option("sql_on", True)
```

```sql
  SELECT
    /*+LABEL('vDataframe._aggregate_matrix')*/ CORR_MATRIX("pclass", "survived", "age", "sibsp", "parch", "fare", "body") OVER ()  
  FROM
(
  SELECT
    RANK() OVER (ORDER BY "pclass") AS "pclass",
    RANK() OVER (ORDER BY "survived") AS "survived",
    RANK() OVER (ORDER BY "age") AS "age",
    RANK() OVER (ORDER BY "sibsp") AS "sibsp",
    RANK() OVER (ORDER BY "parch") AS "parch",
    RANK() OVER (ORDER BY "fare") AS "fare",
    RANK() OVER (ORDER BY "body") AS "body"  
  FROM
"public"."titanic") spearman_table
```

VerticaPy allows users to calculate a focused correlation using the "focus" parameter:

```python
titanic.corr(method="spearman", focus="survived")
```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/c46493b5-61e2-4eca-ae0e-e2a09fc8d304" width="20%">
</p>

- **Data Preparation**

  Whether you are [joining multiple tables](https://www.vertica.com/python/documentation/1.0.x/html/notebooks/data_prep/joins/), [encoding](https://www.vertica.com/python/documentation/1.0.x/html/notebooks/data_prep/encoding/), or [filling missing values](https://www.vertica.com/python/documentation/1.0.x/html/notebooks/data_prep/missing_values/), VerticaPy has everything and more in one package.

```python
import random
import verticapy as vp

data = vp.vDataFrame({"Heights": [random.randint(10, 60) for _ in range(40)] + [100]})
data.outliers_plot(columns="Heights")
```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/c71b106b-29d0-4e19-8267-04c5107aa365" width="50%">
</p>


- **Machine Learning**

  ML is the strongest suite of VerticaPy as it capitalizes on the speed of in-database training and prediction by using SQL in the background to interact with the database. ML for VerticaPy covers a vast array of tools, including [time series forecasting](https://www.vertica.com/python/documentation/1.0.x/html/notebooks/ml/time_series/), [clustering](https://www.vertica.com/python/documentation/1.0.x/html/notebooks/ml/clustering/), and [classification](https://www.vertica.com/python/documentation/1.0.x/html/notebooks/ml/classification/). 

```python
# titanic_vd is already loaded
# Logistic Regression model is already loaded
stepwise_result = stepwise(
    model,
    input_relation=titanic_vd,
    X=[
        "age",
        "fare",
        "parch",
        "pclass",
    ],
    y="survived",
    direction="backward",
    height=600,
    width=800,
)
```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/1550a25c-138c-4673-9940-44bf060a284b" width="50%">
</p>

[:arrow_up: Back to TOC](#table-of-contents)
<br>

### Loading Predefined Datasets

VerticaPy provides some predefined datasets that can be easily loaded. These datasets include the iris dataset, titanic dataset, amazon, and more.

There are two ways to access the provided datasets:

(1) Use the standard python method:

```python
from verticapy.datasets import load_iris

iris_data = load_iris()
```

(2) Use the standard name of the dataset from the public schema:

```python
iris_data = vp.vDataFrame(input_relation = "public.iris")
```
[:arrow_up: Back to TOC](#table-of-contents)
<br>

## Quickstart

The following example follows the <a href='https://www.vertica.com/python/quick-start/'>VerticaPy quickstart guide</a>.

Install the library using with <b>pip</b>.
```shell
root@ubuntu:~$ pip3 install verticapy[all]
```
Create a new Vertica connection:
```python
import verticapy as vp

vp.new_connection({
    "host": "10.211.55.14", 
    "port": "5433", 
    "database": "testdb", 
    "password": "XxX", 
    "user": "dbadmin"},
    name="Vertica_New_Connection")
```
Use the newly created connection:
```python
vp.connect("Vertica_New_Connection")
```
Create a VerticaPy schema for native VerticaPy models (that is, models available in VerticaPy, but not Vertica itself):
```python
vp.create_verticapy_schema()
```
Create a vDataFrame of your relation:
```python
from verticapy import vDataFrame

vdf = vDataFrame("my_relation")
```
Load a sample dataset:
```python
from verticapy.datasets import load_titanic

vdf = load_titanic()
```
Examine your data:
```python
vdf.describe()
```
<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/362dbd53-3692-48e4-a1e1-60f5f565dc50" width="100%">
</p>

Print the SQL query with <b>set_option</b>:
```python
set_option("sql_on", True)
vdf.describe()

# Output
## Compute the descriptive statistics of all the numerical columns ##

SELECT 
  SUMMARIZE_NUMCOL("pclass", "survived", "age", "sibsp", "parch", "fare", "body") OVER ()
FROM public.titanic
```
With VerticaPy, it is now possible to solve a ML problem with few lines of code.
```python
from verticapy.machine_learning.model_selection.model_validation import cross_validate
from verticapy.machine_learning.vertica import RandomForestClassifier

# Data Preparation
vdf["sex"].label_encode()["boat"].fillna(method="0ifnull")["name"].str_extract(
    " ([A-Za-z]+)\."
).eval("family_size", expr="parch + sibsp + 1").drop(
    columns=["cabin", "body", "ticket", "home.dest"]
)[
    "fare"
].fill_outliers().fillna()

# Model Evaluation
cross_validate(
    RandomForestClassifier("rf_titanic", max_leaf_nodes=100, n_estimators=30),
    vdf,
    ["age", "family_size", "sex", "pclass", "fare", "boat"],
    "survived",
    cutoff=0.35,
)
```
<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/49d3a606-8518-4676-b7ae-fa5c3c962432" width="100%">
</p>

```python
# Features importance
model.fit(vdf, ["age", "family_size", "sex", "pclass", "fare", "boat"], "survived")
model.features_importance()
```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/a3d8b236-53a7-4d69-a969-48c2ba9bc114" width="80%">
</p>

```python
# ROC Curve
model = RandomForestClassifier(
    name = "public.RF_titanic",
    n_estimators = 20,
    max_features = "auto",
    max_leaf_nodes = 32, 
    sample = 0.7,
    max_depth = 3,
    min_samples_leaf = 5,
    min_info_gain = 0.0,
    nbins = 32
)
model.fit(
    "public.titanic", # input relation
    ["age", "fare", "sex"], # predictors
    "survived" # response
)

# Roc Curve
model.roc_curve()
```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/87f74bc7-a6cd-4336-8d32-b144f7fb6888" width="80%">
</p>

Enjoy!

[:arrow_up: Back to TOC](#table-of-contents)
<br>

## Help and Support

### Contributing

For a short guide on contribution standards, see the <a href='https://www.vertica.com/python/documentation/1.0.x/html/contribution_guidelines.html'>Contribution Guidelines</a>.

### Communication

- LinkedIn: https://www.linkedin.com/company/verticapy/

- Announcements and Discussion: https://github.com/vertica/VerticaPy/discussions

[:arrow_up: Back to TOC](#table-of-contents)
