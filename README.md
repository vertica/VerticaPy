<p align="center">
<img src='https://raw.githubusercontent.com/vertica/VerticaPy/master/img/logo.png' width="180px">
</p>

# VerticaPy

The documentation is available at:<br>
https://github.com/vertica/VerticaPy/blob/master/documentation.pdf <br><br>
Or directly in the Wiki at:<br>
https://github.com/vertica/VerticaPy/wiki
<br><br>
(c) Copyright [2018-2020] Micro Focus or one of its affiliates. 
Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at
<br>
http://www.apache.org/licenses/LICENSE-2.0
<br>
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
<br><br>
&#9888; If you want to contribute, send a mail to <a href="mailto:badr.ouali@microfocus.com">badr.ouali@microfocus.com</a>
<br><br>
VerticaPy is a Python library that exposes sci-kit like functionality to conduct data science projects on data stored in Vertica, thus taking advantage Vertica’s speed and built-in analytics and machine learning capabilities. It supports the entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize data transformation operation (called Virtual Dataframe), and offers multiple graphical rendering possibilities.
<br><br>
The 'Big Data' (Tb of data) is now one of the main topics in the Data Science World. Data Scientists are now very important for any organisation. Becoming Data-Driven is mandatory to survive. Vertica is the first real analytic columnar Database and is still the fastest in the market. However, SQL is not enough flexible to be very popular for Data Scientists. Python flexibility is priceless and provides to any user a very nice experience. The level of abstraction is so high that it is enough to think about a function to notice that it already exists. Many Data Science APIs were created during the last 15 years and were directly adopted by the Data Science community (examples: pandas and scikit-learn). However, Python is only working in-memory for a single node process. Even if some famous highly distributed programming languages exist to face this challenge, they are still in-memory and most of the time they can not process on all the data. Besides, moving the data can become very expensive. Data Scientists must also find a way to deploy their data preparation and their models. We are far away from easiness and the entire process can become time expensive. 
<br><br>
The idea behind VERTICAPY is simple: Combining the Scalability of VERTICA with the Flexibility of Python to give to the community what they need *Bringing the logic to the data and not the opposite*. This version 1.0 is the work of 3 years of new ideas and improvement.
<br><br>
Main Advantages:
<ul>
 <li> easy Data Exploration.</li>
 <li> easy Data Preparation.</li>
 <li> easy Data Modeling.</li>
 <li> easy Model Evaluation.</li>
 <li> easy Model Deployment.</li>
 <li> most of what pandas.Dataframe can do, verticapy.vDataframe can do (and even much more)</li>
 <li> easy ML model creation and evaluation.</li>
 <li> many scikit functions and algorithms are available (and scalable!).</li>
</ul>

All information related to the API can be found at: <br>

https://github.com/vertica/VerticaPy/

## Python Version

<b>VerticaPy</b> works with at least:
<ul>
	<li> <b>Vertica: </b> => 9.1 (with previous versions, some functions and algorithms may not be available)
	<li> <b>Python Version: </b> => 3.6 - [version 3.5 may works]
	<li> <b>Python Modules needed for Rendering Capabilities: </b> matplotlib (=> 2.0) - [other versions of matplotlib may work], numpy (=> 1.9) - [other versions of numpy may work]
	<li> <b>Other Python Modules: </b> Except to get rendering capabilities, VerticaPy uses only built-in libraries (statistics, random, math, time and os)
</ul>

## Standard Libraries

<b>VerticaPy</b> library is only using the standard Python libraries such as <b>matplotlib</b>, <b>numpy</b>... 
Other libraries can be used as <b>anytree</b> for tree visualization or <b>sqlparse</b> for SQL indentation but they are optional.

## Installation
To install <b>VerticaPy</b>, you can use the pip command:
```shell
root@ubuntu:~$ pip3 install verticapy
```
Or you can get a copy of the source by cloning from the VerticaPy github project and install with:
```shell
root@ubuntu:~$ python3 setup.py install
```
You can also drag and drop the <b>verticapy</b> folder in the <b>site-package</b> folder of the Python framework. In the MAC environment, you can find it in: <br> <b>/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages</b> <br>

Another way is to call the library from where it is located. <br>

You can then import each library element using the usual Python syntax.
```python
# to import the vDataframe
from verticapy import vDataframe
# to import the Logistic Regression
from verticapy.learn.linear_model import LogisticRegression
```

Everything is well detailed in the following documentation.

## Connection to the Database

This step is useless if <b>vertica-python</b> or <b>pyodbc</b> is already installed and you have a DSN in your machine. With this configuration, you do not need to manually create a cursor. It is possible to create a vDataframe using directly the DSN (<b>dsn</b> parameter of the vDataframe).

### ODBC

To connect to the database, the user can use an ODBC connection to the Vertica database. <b>vertica-python</b> and <b>pyodbc</b> provide a cursor that will point to the database. It will be used by the <b>VerticaPy</b> to create all the different objects.
```python
#
# vertica_python
#
import vertica_python

# Connection using all the DSN information
conn_info = {'host': "10.211.55.14", 'port': 5433, 'user': "dbadmin", 'password': "XxX", 'database': "testdb"}
cur = vertica_python.connect(** conn_info).cursor()

# Connection using directly the DSN
from verticapy.utilities import to_vertica_python_format # This function will parse the odbc.ini file
dsn = "VerticaDSN"
cur = vertica_python.connect(** to_vertica_python_format(dsn)).cursor()

#
# pyodbc
#
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

The user can also use a JDBC connection to the Vertica Database. 
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

Install the library using the <b>pip</b> command:
```shell
root@ubuntu:~$ pip3 install verticapy
```
Install <b>vertica_python</b> or <b>pyodbc</b> to build a DB cursor:
```shell
root@ubuntu:~$ pip3 install vertica_python
```
Create a vertica cursor
```python
from verticapy.utilities import vertica_cursor
cur = vertica_cursor("VerticaDSN")
```
Create the Virtual Dataframe of your relation:
```python
from verticapy import vDataframe
vdf = vDataframe("my_relation", cursor = cur)
```
If you don't have data to play, you can easily load well known datasets
```python
from verticapy.learn.datasets import load_titanic
vdf = load_titanic(cursor = cur)
```
You can now play with the data...
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

You can also print the SQL code generation using the <b>sql_on_off</b> method.

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
cross_validate(RandomForestClassifier("rf_titanic", cur, max_leaf_nodes = 100, n_estimators = 30), "titanic_clean", ["age", "family_size", "sex", "pclass", "fare", "boat"], "survived", cutoff = 0.35)

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

