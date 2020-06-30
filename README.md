<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/logo.png' width="180px">
</p>

:loudspeaker: 2020-06-27: VerticaPy is the new name for Vertica-ML-Python.

# VerticaPy

[![PyPI version](https://badge.fury.io/py/verticapy.svg)](https://badge.fury.io/py/verticapy)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/verticapy.svg)](https://www.python.org/downloads/)

VerticaPy is a Python library that exposes sci-kit like functionality to conduct data science projects on data stored in Vertica, thus taking advantage Vertica’s speed and built-in analytics and machine learning capabilities. It supports the entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize data transformation operation (called Virtual DataFrame), and offers multiple graphical rendering possibilities.
<br><br>
The 'Big Data' (+100 Tb of data) is now one of the main topics in the Data Science World. Data Scientists are now very important for any organisation. Becoming Data-Driven is mandatory to survive. Vertica is the first real analytic columnar Database and is still the fastest in the market. However, SQL is not enough flexible to be very popular for Data Scientists. Python flexibility is priceless and provides to any user a very nice experience. The level of abstraction is so high that it is enough to think about a function to notice that it already exists. Many Data Science APIs were created during the last 15 years and were directly adopted by the Data Science community (examples: pandas and scikit-learn). However, Python is only working in-memory for a single node process. Even if some famous highly distributed programming languages exist to face this challenge, they are still in-memory and most of the time they can not process on all the data. Besides, moving the data can become very expensive. Data Scientists must also find a way to deploy their data preparation and their models. We are far away from easiness and the entire process can become time expensive. 
<br><br>
The idea behind VerticaPy is simple: Combining the Scalability of VERTICA with the Flexibility of Python to give to the community what they need *Bringing the logic to the data and not the opposite*. This version 1.0 is the work of 3 years of new ideas and improvement.
<br><br>
Main Advantages:
<ul>
 <li> Easy Data Exploration.</li>
 <li> Fast Data Preparation.</li>
 <li> In DataBase Machine Learning.</li>
 <li> Easy Model Evaluation.</li>
 <li> Easy Model Deployment.</li>
</ul>

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/architecture.png' width="80%">
</p>

## Installation
To install <b>VerticaPy</b>, you can use the following pip command.
```shell
root@ubuntu:~$ pip3 install verticapy
```
Or you can get a copy of the source by cloning from the VerticaPy github project and install it using the following command.
```shell
root@ubuntu:~$ python3 setup.py install
```

## Documentation

A well-detailed HTML documentation is available by downloading the zip file at: <br>

https://github.com/vertica/VerticaPy/blob/master/documentation.zip

## Connection to the Database

To connect to the database, many options are available.

### Native Client

```python
import vertica_python

# Connection using all the DSN information
conn_info = {'host': "10.211.55.14", 'port': 5433, 'user': "dbadmin", 'password': "XxX", 'database': "testdb"}
cur = vertica_python.connect(** conn_info).cursor()

# Connection using directly the DSN
from verticapy.utilities import to_vertica_python_format # This function will parse the odbc.ini file
dsn = "VerticaDSN"
cur = vertica_python.connect(** to_vertica_python_format(dsn)).cursor()
```

### ODBC

```python
import pyodbc

# Connection using all the DSN information
driver = "/Library/Vertica/ODBC/lib/libverticaodbc.dylib"
server = "10.211.55.14"
database = "testdb"
port = "5433"
uid = "dbadmin"
pwd = "XxX"
dsn = ("DRIVER={}; SERVER={}; DATABASE={}; PORT={}; UID={}; PWD={};").format(driver, server, database, port, uid, pwd)
cur = pyodbc.connect(dsn).cursor()

# Connection using directly the DSN
dsn = ("DSN=VerticaDSN")
cur = pyodbc.connect(dsn).cursor()
```

### JDBC
 
```python
import jaydebeapi

# Vertica Server Details
database = "testdb"
hostname = "10.211.55.14"
port = "5433"
uid = "dbadmin"
pwd = "XxX"

# Vertica JDBC class name
jdbc_driver_name = "com.vertica.jdbc.Driver"

# Vertica JDBC driver path
jdbc_driver_loc = "/Library/Vertica/JDBC/vertica-jdbc-9.3.1-0.jar"

# JDBC connection string
connection_string = 'jdbc:vertica://' + hostname + ':' + port + '/' + database
url = '{}:user={};password={}'.format(connection_string, uid, pwd)
conn = jaydebeapi.connect(jdbc_driver_name, connection_string, {'user': uid, 'password': pwd}, jars = jdbc_driver_loc)
cur = conn.cursor()
```

## Quick Start

Install the library using the <b>pip</b> command.
```shell
root@ubuntu:~$ pip3 install verticapy
```
Install <b>vertica_python</b> or <b>pyodbc</b> to create a DB cursor.
```shell
root@ubuntu:~$ pip3 install vertica_python
```
Create a vertica cursor.
```python
from verticapy import vertica_cursor
cur = vertica_cursor("VerticaDSN")
```
Create the Virtual DataFrame of your relation.
```python
from verticapy import vDataFrame
vdf = vDataFrame("my_relation", cursor = cur)
```
If you don't have data to play, you can easily load well known datasets.
```python
from verticapy.learn.datasets import load_titanic
vdf = load_titanic(cursor = cur)
```
You can play with the data...
```python
vdf.describe()

# Output
               min       25%        50%        75%   
age           0.33      21.0       28.0       39.0   
body           1.0     79.25      160.5      257.5   
fare           0.0    7.8958    14.4542    31.3875   
parch          0.0       0.0        0.0        0.0   
pclass         1.0       1.0        3.0        3.0   
sibsp          0.0       0.0        0.0        1.0   
survived       0.0       0.0        0.0        1.0   
                   max    unique  
age               80.0        96  
body             328.0       118  
fare          512.3292       277  
parch              9.0         8  
pclass             3.0         3  
sibsp              8.0         7  
survived           1.0         2 
```
It is also possible to print the SQL code generation using the <b>sql_on_off</b> method.
```python
vdf.sql_on_off()
vdf.describe()

# Output
## Compute the descriptive statistics of all the numerical columns ##

SELECT 
  SUMMARIZE_NUMCOL("age","body","survived","pclass","parch","fare","sibsp") OVER ()
FROM public.titanic
```
With VerticaPy, it is now possible to solve a ML problem with few lines of code.
```python
from verticapy.learn.model_selection import cross_validate
from verticapy.learn.ensemble import RandomForestClassifier

# Data Preparation
vdf["sex"].label_encode()["boat"].fillna(method = "0ifnull")["name"].str_extract(' ([A-Za-z]+)\.').eval("family_size", expr = "parch + sibsp + 1").drop(columns = ["cabin", "body", "ticket", "home.dest"])["fare"].fill_outliers().fillna().to_db("titanic_clean")

# Model Evaluation
cross_validate(RandomForestClassifier("rf_titanic", cur, max_leaf_nodes = 100, n_estimators = 30), 
               "titanic_clean", 
               ["age", "family_size", "sex", "pclass", "fare", "boat"], 
               "survived", 
               cutoff = 0.35)

# Output
                           auc               prc_auc   
1-fold      0.9877114427860691    0.9530465915039339   
2-fold      0.9965555014605642    0.7676485351425721   
3-fold      0.9927239216549301    0.6419135521132449   
avg             0.992330288634        0.787536226253   
std           0.00362128464093         0.12779562393   
                     accuracy              log_loss   
1-fold      0.971291866028708    0.0502052541223871   
2-fold      0.983253588516746    0.0298167751798457   
3-fold      0.964824120603015    0.0392745694400433   
avg            0.973123191716       0.0397655329141   
std           0.0076344236729      0.00833079837099   
                     precision                recall   
1-fold                    0.96                  0.96   
2-fold      0.9556962025316456                   1.0   
3-fold      0.9647887323943662    0.9383561643835616   
avg             0.960161644975        0.966118721461   
std           0.00371376912311        0.025535200301   
                      f1-score                   mcc   
1-fold      0.9687259282082884    0.9376119402985075   
2-fold      0.9867172675521821    0.9646971010878469   
3-fold      0.9588020287309097    0.9240569687684576   
avg              0.97141507483        0.942122003385   
std            0.0115538960753       0.0168949813163   
                  informedness            markedness   
1-fold      0.9376119402985075    0.9376119402985075   
2-fold      0.9737827715355807    0.9556962025316456   
3-fold      0.9185148945422918    0.9296324823943662   
avg             0.943303202125        0.940980208408   
std            0.0229190954261       0.0109037699717   
                           csi  
1-fold      0.9230769230769231  
2-fold      0.9556962025316456  
3-fold      0.9072847682119205  
avg             0.928685964607  
std            0.0201579224026
```
Happy Coding !