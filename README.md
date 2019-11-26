
<p align="center">
<img src='./img/logo.png' width="180px">
</p>
<p align="center">
<i>Scalable as Vertica, Flexible as Python</i>
</p>
The documentation is available at: 
https://github.com/vertica/Vertica-ML-Python/blob/master/documentation.pdf 
Or directly in the Wiki at:
https://github.com/vertica/Vertica-ML-Python/wiki
<br><br>
(c) Copyright [2018] Micro Focus or one of its affiliates. 
Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

&#9888; If you want to contribute, send a mail to <a href="mailto:badr.ouali@microfocus.com">badr.ouali@microfocus.com</a>

# Vertica-ML-Python

Vertica-ML-Python is a Python library that exposes sci-kit like functionality to conduct data science projects on data stored in Vertica, thus taking advantage Vertica’s speed and built-in analytics and machine learning capabilities. It supports the entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize data transformation operation (called Virtual Dataframe), and offers multiple graphical rendering possibilities.

## Features

Scalability of Vertica and Flexibility of Python:

<ul>
  <li>Use only standard libraries (except for trees drawing)</li>
  <li>Use JDBC or ODBC connections</li>
  <li>Data Exploration (Histograms, Pie/Donut Charts, 2D/3D Scatter Plots, Correlation Matrix...)</li>
  <li>Data Preparation (Normalization, One Hot Encoding, Label Encoding, Imputing Missing Values...)</li>
  <li>Multiple Machine Learning Algorithms (Logistic Regression, Random Forest, SVM, Naive Bayes, KMeans...)</li>
  <li>Models Evaluation (Cross Validation, ROC curve, accuracy, logloss, Lift Table...)</li>
</ul>

## Why using this Library?

The 'Big Data' (Tb of data) is now one of the main topics in the Data Science World. Data Scientists are now very important for any organisation. Becoming Data-Driven is mandatory to survive. Vertica is the first real analytic columnar Database and is still the fastest in the market. However, SQL is not enough flexible to be very popular for Data Scientists. Python flexibility is priceless and provides to any user a very nice experience. The level of abstraction is so high that it is enough to think about a function to notice that it already exists. Many Data Science APIs were created during the last 15 years and were directly adopted by the Data Science community (examples: pandas and scikit-learn). However, Python is only working in-memory for a single node process. Even if some famous highly distributed programming languages exist to face this challenge, they are still in-memory and most of the time they can not process on all the data. Besides, moving the data can become very expensive. Data Scientists must also find a way to deploy their data preparation and their models. We are far away from easiness and the entire process can become time expensive. 

The idea behind VERTICA ML PYTHON is simple: Combining the Scalability of VERTICA with the Flexibility of Python to give to the community what they need *Bringing the logic to the data and not the opposite*. This version 1.0 is the work of 3 years of new ideas and improvement.

Main Advantages:
 - easy Data Exploration.
 - easy Data Preparation.
 - easy Data Modeling.
 - easy Model Evaluation.
 - easy Model Deployment.
 - most of what pandas.Dataframe can do, vertica_ml_python.vDataframe can do (and sometimes even more)
 - easy ML model creation and evaluation.
 - many scikit functions and algorithms are available (and scalable!).

&#9888; Please read the Vertica ML Python Documentation. If you do not have time just read below.

&#9888; The previous API is really nothing compare to the new version and many methods and functions totally changed. Consider this API as a totally new one.

If you have any feedback about the library, please contact me: <a href="mailto:badr.ouali@microfocus.com">badr.ouali@microfocus.com</a>

## Prerequires

<b>vertica-ml-python</b> works with at least:
<ul>
	<li> <b>Vertica: </b> 9.1 (with previous versions, some functions and algorithms may not be available)
	<li> <b>Python: </b> 3.6 
</ul>

## Installation:

To install <b>vertica-ml-python</b>, you can use the pip command:
```
root@ubuntu:~$ pip install vertica_ml_python
```
You can also drag and drop the <b>vertica_ml_python</b> folder in the <b>site-package</b> folder of the Python framework. In the MAC environment, you can find it in: <br> <b>/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages</b> <br>

Another way is to call the library from where it is located. <br>

You can then import each library element using the usual Python syntax.
```
# to import the vDataframe
from vertica_ml_python import vDataframe
# to import the Logistic Regression
from vertica_ml_python.learn.linear_model import LogisticRegression
```

## Quick Start

Install the library using the <b>pip</b> command:
```
root@ubuntu:~$ pip install vertica_ml_python
```
Install <b>vertica_python</b> or <b>pyodbc</b> to build a DB cursor:
```
pip install vertica_python
```
Create a vertica cursor
```
from vertica_ml_python.utilities import vertica_cursor
cur = vertica_cursor("VerticaDSN")
```
Create the Virtual Dataframe of your relation:
```
from vertica_ml_python import vDataframe
vdf = vDataframe("my_relation", cursor = cur)
```
If you don't have data to play, you can easily load well known datasets
```
from vertica_ml_python.learn.datasets import load_titanic
vdf = load_titanic(cursor = cur)
```
You can now play with the data...
```
vdf.describe()

# Output
               min       25%        50%        75%   <br>
age           0.33      21.0       28.0       39.0   <br>
body           1.0     79.25      160.5      257.5   <br>
fare           0.0    7.8958    14.4542    31.3875   <br>
parch          0.0       0.0        0.0        0.0   <br>
pclass         1.0       1.0        3.0        3.0   <br>
sibsp          0.0       0.0        0.0        1.0   <br>
survived       0.0       0.0        0.0        1.0   <br>
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

```
vdf.sql_on_off()
vdf.describe()

# Output
<i> Compute the descriptive statistics of all the numerical columns </i>

SELECT SUMMARIZE_NUMCOL("age","body","survived","pclass","parch","fare","sibsp") OVER ()
FROM
  (SELECT "age" AS "age",
          "body" AS "body",
          "survived" AS "survived",
          "ticket" AS "ticket",
          "home.dest" AS "home.dest",
          "cabin" AS "cabin",
          "sex" AS "sex",
          "pclass" AS "pclass",
          "embarked" AS "embarked",
          "parch" AS "parch",
          "fare" AS "fare",
          "name" AS "name",
          "boat" AS "boat",
          "sibsp" AS "sibsp"
   FROM public.titanic) final_table
```

With Vertica ML Python, it is now possible to solve a ML problem with four lines of code (two if we don't consider the libraries loading).

```
from vertica_ml_python.learn.model_selection import cross_validate
from vertica_ml_python.learn.linear_model import LogisticRegression

# Data Preparation
vdf["sex"].label_encode().parent["boat"].fillna(method = "0ifnull").parent["name"].str_extract(' ([A-Za-z]+)\.').parent.eval("family_size", expr = "parch + sibsp + 1").drop(columns = ["cabin", "body", "ticket", "home.dest"])["fare"].fill_outliers().parent.fillna().to_db("titanic_clean")

# Model Evaluation
cross_validate(RandomForestClassifier("logit_titanic", cur, max_leaf_nodes = 100, n_estimators = 30), "titanic_clean", ["age", "family_size", "sex", "pclass", "fare", "boat"], "survived", cutoff = 0.35)

# Output
                           auc               prc_auc   <br>
1-fold      0.9877114427860691    0.9530465915039339   <br>
2-fold      0.9965555014605642    0.7676485351425721   <br>
3-fold      0.9927239216549301    0.6419135521132449   <br>
avg             0.992330288634        0.787536226253   <br>
std           0.00362128464093         0.12779562393   <br>
                     accuracy              log_loss   <br>
1-fold      0.971291866028708    0.0502052541223871   <br>
2-fold      0.983253588516746    0.0298167751798457   <br>
3-fold      0.964824120603015    0.0392745694400433   <br>
avg            0.973123191716       0.0397655329141   <br>
std           0.0076344236729      0.00833079837099   <br>
                     precision                recall   <br>
1-fold                    0.96                  0.96   <br>
2-fold      0.9556962025316456                   1.0   <br>
3-fold      0.9647887323943662    0.9383561643835616   <br>
avg             0.960161644975        0.966118721461   <br>
std           0.00371376912311        0.025535200301   <br>
                      f1-score                   mcc   <br>
1-fold      0.9687259282082884    0.9376119402985075   <br>
2-fold      0.9867172675521821    0.9646971010878469   <br>
3-fold      0.9588020287309097    0.9240569687684576   <br>
avg              0.97141507483        0.942122003385   <br>
std            0.0115538960753       0.0168949813163   <br>
                  informedness            markedness   <br>
1-fold      0.9376119402985075    0.9376119402985075   <br>
2-fold      0.9737827715355807    0.9556962025316456   <br>
3-fold      0.9185148945422918    0.9296324823943662   <br>
avg             0.943303202125        0.940980208408   <br>
std            0.0229190954261       0.0109037699717   <br>
                           csi  
1-fold      0.9230769230769231  
2-fold      0.9556962025316456  
3-fold      0.9072847682119205  
avg             0.928685964607  
std            0.0201579224026
```

Happy Playing !

