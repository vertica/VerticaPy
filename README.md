<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/logo.png' width="180px">
</p>

:loudspeaker: 2020-06-27: VerticaPy is the new name for Vertica-ML-Python.

# VerticaPy

[![PyPI version](https://badge.fury.io/py/verticapy.svg)](https://badge.fury.io/py/verticapy)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/pypi/pyversions/verticapy.svg)](https://www.python.org/downloads/)

VerticaPy is a Python library with scikit-like functionality used to conduct data science projects on data stored in Vertica, taking advantage Vertica’s speed and built-in analytics and machine learning features. It supports the entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize data transformation operations, and offers beautiful graphical options.
<br><br>
Nowadays, 'Big Data' is one of the main topics in the data science world, and data scientists are often at the center of any organization. The benefits of becoming more data-driven are undeniable and are often needed to survive in the industry.
<br><br>
Vertica was the first real analytic columnar database and is still the fastest in the market. However, SQL alone isn't flexible enough to meet the needs of data scientists.
<br><br>
Python has quickly become the most popular tool in this domain, owing much of its flexibility to its high-level of abstraction and impressively large and ever-growing set of libraries. Its accessibility has led to the development of popular and perfomant APIs, like pandas and scikit-learn, and a dedicated community of data scientists. Unfortunately, Python only works in-memory as a single-node process. This problem has led to the rise of distributed programming languages, but they too, are limited as in-memory processes and, as such, will never be able to process all of your data in this era, and moving data for processing is prohobitively expensive. On top of all of this, data scientists must also find convenient ways to deploy their data and models. The whole process is time consuming.
<br><br>
**VerticaPy aims to solve all of these problems**. The idea is simple: instead of moving data around for processing, VerticaPy brings the logic to the data.
<br><br>
3 years in the making, we're proud to bring you VerticaPy.
<br><br>
Main Advantages:
<ul>
 <li> Easy Data Exploration.</li>
 <li> Fast Data Preparation.</li>
 <li> In-Database Machine Learning.</li>
 <li> Easy Model Evaluation.</li>
 <li> Easy Model Deployment.</li>
</ul>

<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/architecture.png' width="92%">
</p>

## Installation
To install <b>VerticaPy</b> with pip:
```shell
# Latest release version
root@ubuntu:~$ pip3 install verticapy

# Latest commit on master branch
root@ubuntu:~$ pip3 install git+https://github.com/vertica/verticapy.git@master
```
To install <b>VerticaPy</b> from source, run the following command from the root directory:
```shell
root@ubuntu:~$ python3 setup.py install
```

## Documentation

A well-detailed HTML documentation is available at: <br>

https://www.vertica.com/python/documentation_last/

## Connecting to the Database

VerticaPy is compatible with several clients.

### Native Client (Recommended)

```python
import vertica_python

# Connection using all the DSN information
conn_info = {'host': "10.211.55.14", 
             'port': 5433, 
             'user': "dbadmin", 
             'password': "XxX", 
             'database': "testdb"}
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
Create a vertica cursor.
```python
from verticapy import vertica_conn
cur = vertica_conn("VerticaDSN").cursor()
```
Create the Virtual DataFrame of your relation.
```python
from verticapy import vDataFrame
vdf = vDataFrame("my_relation", cursor = cur)
```
If you don't have data on hand, you can easily import well-known datasets.
```python
from verticapy.learn.datasets import load_titanic
vdf = load_titanic(cursor = cur)
```
Examine your data:
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
Print the SQL query with the <b>set_option</b> function:
```python
set_option("sql_on", True)
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
vdf["sex"].label_encode()["boat"].fillna(method = "0ifnull")["name"].str_extract(' ([A-Za-z]+)\.').eval("family_size", expr = "parch + sibsp + 1").drop(columns = ["cabin", "body", "ticket", "home.dest"])["fare"].fill_outliers().fillna()

# Model Evaluation
cross_validate(RandomForestClassifier("rf_titanic", cur, max_leaf_nodes = 100, n_estimators = 30), 
               vdf, 
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
Enjoy!
