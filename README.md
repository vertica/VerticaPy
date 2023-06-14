<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/logo.png' width="180px">
</p>

:star: 2022-12-01: VerticaPy secures 100 stars.

:loudspeaker: 2020-06-27: Vertica-ML-Python has been renamed to VerticaPy.

:warning: VerticaPy 0.9.0 (and onwards) includes several significant changes and is therefore not backward compatible with older versions. For details, see the <a href='https://www.vertica.com/python/documentation_last/whats-new.php'>changelog</a>.

:warning: The following README is for VerticaPy 1.0.0-beta and onwards, and so some of the elements may not be present in the previous versions. 

# VerticaPy

[![PyPI version](https://badge.fury.io/py/verticapy.svg)](https://badge.fury.io/py/verticapy)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/verticapy?color=yellowgreen)](https://anaconda.org/conda-forge/verticapy)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/verticapy.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/vertica/VerticaPy/branch/master/graph/badge.svg?token=a6GiFYI9at)](https://codecov.io/gh/vertica/VerticaPy)

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/benefits.png' width="92%">
</p>

VerticaPy is a Python library with scikit-like functionality used to conduct data science projects on data stored in Vertica, taking advantage of Vertica’s speed and built-in analytics and machine learning features. VerticaPy offers robust support for the entire data science life cycle, uses a 'pipeline' mechanism to sequentialize data transformation operations, and offers beautiful graphical options.
<br><br>

# Table on Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Connecting to the Database](#connecting-to-the-database)
- [Documentation](#documentation)
- [Use-cases](#use-cases)
- [Highlighted Features](#highllighted-features)
  - [SQL Magic](#sql-magic)
  - [SQL Plots](#sql-plots)
  - [Diverse Database Connections](#multiple-database-connection-using-dblink)
  - [Python and SQL Combo](#python-and-sql-combo)
  - [Charts](#charts)
  - [Complete ML pipeline](#compelte-machine-learning-pipeline)
- [Quickstart](#quickstart)
- [Help and Support](#help-an-support)
  - [Contributing](#contributing)
  - [Communication](#communication)

<br><br>
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
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/architecture.png' width="92%">
</p>

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

https://www.vertica.com/python/installation.php

## Connecting to the Database

VerticaPy is compatible with several clients. For details, see the <a href='https://www.vertica.com/python/connection.php'>connection page</a>.<br>

## Documentation

The easiest and most accurate way to find documentaiton for a particular function is to use the help function:

```python
import verticapy as vp
help(vp.vDataFrame)
```

Official documentation is available at: <br>

https://www.vertica.com/python/documentation_last/

:heavy_exclamation_mark: But note the above is not currently updated as per VerticaPy 1.0.0-beta. It will be done soon.

## Use-cases

Examples and case-studies: <br>

https://www.vertica.com/python/examples/

<p align="center">
<img src="https://raw.githubusercontent.com/vertica/VerticaPy/master/img/examples.gif" width="92%">
</p>

## Highlighted Features

### SQL Magic
You can use VerticaPy to execute SQL queries directly from a Jupyter notebook. For details, see <a href='https://www.vertica.com/python/documentation_last/extensions/sql/'>SQL Magic</a>:

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
### SQL Plots

You can create interactive, professional plots directly from SQL.

To create plots, simply provide the type of plot along with the SQL command.

#### Example
```python
%load_ext verticapy.jupyter.extensions.chart_magic
%chart -k pie -c "SELECT pclass, AVG(age) AS av_avg FROM titanic GROUP BY 1;"
```
<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/93e8556d-851d-4f1d-8f75-e46e9feed65a" width="50%">
</p>


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

For more details on how to setup DBLINK, please visit the [github repo](https://github.com/vertica/dblink). To learn about using DBLINK in VerticaPy, check out the [documentation page](https://www.vertica.com/python/workshop/full_stack/dblink_integration/index.php).


### Python and SQL Combo

VerticaPy has a unique place in the market because it allows users to use python and SQL in the same environment. 

#### Example
```python
import verticapy as vp
selected_titanic = vp.vDataFrame(
    "(SELECT pclass, embarked, AVG(survived) FROM public.titanic GROUP BY 1, 2) x"
)
selected_titanic.groupby(columns=["pclass"], expr=["AVG(AVG)"])
```

### Charts

Verticapy comes integrated with three popular plotting libraries: matplotlib, highcharts, and plotly.

A gallery of VerticaPy-generated charts is available at:<br>

https://www.vertica.com/python/gallery/

<p align="center">
<img src="https://raw.githubusercontent.com/vertica/VerticaPy/master/img/charts.gif" width="92%">
</p>

### Compelte Machine Learning Pipeline

- **Data Ingestion**

  VerticaPy allows users to ingest data from a diverse range of sources, such as AVRO, Parquet, CSV, JSON etc. With a simple command "[read_file](https://www.vertica.com/python/documentation_last/utilities/read_file/)", VerticaPy automatically infers the source type and the data type.

  ```python
  import verticapy as vp
  vp.read_file(path="complex.csv")
  ```

- **Data Exploration**

  There are many options for descriptive and visual exploration. 

```python
from verticapy.datasets import load_iris
iris_data = load_iris()
iris_data.scatter(
    ["SepalWidthCm", "SepalLengthCm", "PetalLengthCm"], by="Species", max_nb_points=30
)
```
<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/ffa37b72-2778-4ea5-af9e-c0f3d6f610f3" width="40%">
</p>

- **Data Preparation**

  Whether you are [joining multiple tables](https://www.vertica.com/python/workshop/data_prep/joins/), [encoding](https://www.vertica.com/python/workshop/data_prep/encoding/index.php), or [filling missing values](https://www.vertica.com/python/workshop/data_prep/missing_values/index.php), VerticaPy has everything and more in one package.

```python
import random
import verticapy as vp
data = vp.vDataFrame({"Heights": [random.randint(10, 60) for _ in range(40)] + [100]})
data.outliers_plot(columns="Heights")
```
<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/ff5cf6fa-b54b-4910-bc11-60305fcbc871" width="40%">
</p>


- **Machine Learning**

  ML is the strongest suite of VerticaPy as it capitalizes on the speed of in-database training and prediction by using SQL in the background to interact with the database. ML for VerticaPy covers a vast array of tools, including [time series forecasting](https://www.vertica.com/python/workshop/ml/time_series/index.php), [clustering](https://www.vertica.com/python/workshop/ml/clustering/index.php), and [classification](https://www.vertica.com/python/workshop/ml/classification/index.php). 

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
<img src="https://github.com/vertica/VerticaPy/assets/46414488/aaefb9bc-9825-4f31-b411-b2ef06a8bed7" width="50%">
</p>


## Quickstart

The following example follows the <a href='https://www.vertica.com/python/quick-start.php'>VerticaPy quickstart guide</a>.

Install the library using with <b>pip</b>.
```shell
root@ubuntu:~$ pip3 install verticapy[all]
```
Create a new Vertica connection:
```python
import verticapy as vp
vp.new_connection({"host": "10.211.55.14", 
                   "port": "5433", 
                   "database": "testdb", 
                   "password": "XxX", 
                   "user": "dbadmin"},
                   name = "Vertica_New_Connection")
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
from verticapy.learn.model_selection import cross_validate
from verticapy.learn.ensemble import RandomForestClassifier

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
<img src="https://github.com/vertica/VerticaPy/assets/46414488/81788c21-b16d-4b41-8cba-5ffa47f6ef04" width="80%">
</p>

```python
# ROC Curve
model.roc_curve(pos_label="0")
```

<p align="center">
<img src="https://github.com/vertica/VerticaPy/assets/46414488/91336750-56b0-4984-8d0f-33f7b22a2f41" width="80%">
</p>


Enjoy!

## Help and Support

### Contributing

For a short guide on contribution standards, see <a href='https://github.com/vertica/VerticaPy/blob/master/CONTRIBUTING.md'>CONTRIBUTING.md</a>.

### Communication

- LinkedIn: https://www.linkedin.com/company/verticapy/

- Announcements and Discussion: https://github.com/vertica/VerticaPy/discussions
