
<p align="center">
<img src='./notebooks/images/vpython.png' width="230px">
</p>

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

&#9888; If you want to contribute, you can post your notebook in the 'notebooks' folder! For more information, send a mail to <a href="mailto:badr.ouali@microfocus.com">badr.ouali@microfocus.com</a>

To see the available notebooks, please take a look at this link: http://nbviewer.jupyter.org/github/vertica/vertica_ml_python/

# Vertica-ML-Python

Vertica-ML-Python is a Python library that exposes sci-kit like functionality to conduct data science projects on data stored in Vertica, thus taking advantage Vertica’s speed and built-in analytics and machine learning capabilities. It supports the entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize data transformation operation (called Resilient Vertica Dataset), and offers multiple graphical rendering possibilities.

## Features

Everything can be done with only one line of code.

<ul>
  <li>Use only standard libraries (except for trees drawing)</li>
  <li>Use JDBC or ODBC connections</li>
  <li>Data Visualization (Histograms, Pie/Donut Charts, 2D/3D Scatter Plots, Correlation Matrix...)</li>
  <li>Data Preparation (Normalization, One Hot Encoding, Label Encoding, Imputing Missing Values...)</li>
  <li>Multiple Machine Learning Algorithms (Logistic Regression, Random Forest, SVM, Naive Bayes, Kmeans...)</li>
  <li>Models Evaluation (Cross Validation, ROC curve, accuracy, logloss, Lift Table...)</li>
</ul>


## Why using this Library?

When we deal with Big Data, it is quite hard to find an API which can be flexible and easy to use. A lot of platforms offer the possibility to create datasets/dataframes which represent just a sample of our data. Most of the time, the data is loaded in memory which is a quite limited method. However, we need a real adaptation when we want to use all our data and do not move it.

Vertica-ML-Python allows users to use some simple Python methods to solve the problem using Vertica. Many objects having very easy methods are available to make the datascience journey exciting. It looks like a Data Science Studio for programmers in the use but without the inconvenient to load data in memory. The user can then explore all the data he has, do all the data preparation and create a model without modifying anything (or even loading the data!). Vertica-ML-Python will help him to generate the SQL pipeline he needs to create the object. 

For example, to describe the titanic dataset Vertica-ML-Python will send to Vertica (using ODBC or JDBC connection) the following query.

```
select summarize_numcol(age,body,fare,parch,pclass,sibsp,survived) over ()
from
  (select *
   from
     (select age as age,
             boat as boat,
             body as body,
                     cabin as cabin,
                     embarked as embarked,
                     fare as fare,
                     homedest as homedest,
                     name as name,
                     parch as parch,
                     pclass as pclass,
                     sex as sex,
                     sibsp as sibsp,
                     survived as survived,
                     ticket as ticket
      from titanic
      offset 0) t1) new_table
```

Everything is generated using a really simple method.

```
titanic.describe()

# Output
              count                 mean                  std     min   \\
age            1046      29.881137667304     14.4134932112713    0.17   \\
body            121     160.809917355372     97.6969219960031     1.0   \\
fare           1308     33.2954792813456     51.7586682391741     0.0   \\
parch          1309    0.385026737967914    0.865560275349515     0.0   \\
pclass         1309     2.29488158899923    0.837836018970128     1.0   \\
sibsp          1309    0.498854087089381      1.0416583905961     0.0   \\
survived       1309    0.381970970206264    0.486055170866483     0.0   \\
                 25%        50%       75%         max    cardinality  
age             21.0       28.0      39.0        80.0             98  
body            72.0      155.0     256.0       328.0            121  
fare          7.8958    14.4542    31.275    512.3292            281  
parch            0.0        0.0       0.0         9.0              8  
pclass           2.0        3.0       3.0         3.0              3  
sibsp            0.0        0.0       1.0         8.0              7  
survived         0.0        0.0       1.0         1.0              2 
```

To impute a column, it will just save inside the object the correct 'coalesce' statement. 
To encode a feature using a label encoding, it will simply save inside the object the correct 'decode' statement.

It introduces an object called the RVD which is quite similar in the use to pandas.Dataframe in order to make the Python users feel comfortable. The following example shows how to create a RVD from a csv file (the titanic dataset) and draw the 'embarked' feature histogram.

```
# Creation of the pyodbc cursor
import pyodbc
cur=pyodbc.connect("DSN=VerticaDSN").cursor()

# Creation of the RVD from a csv file
from vertica_ml_python import read_csv
titanic=read_csv('titanic.csv',cur)

titanic["embarked"].hist()
```
<p align="center">
<img src='./notebooks/images/embarked_hist.png' width="480px">
</p>

The following example shows how to create a logistic regression model and evaluate it.

```
# Creation of the logistic regression model
from vertica_ml_python import logistic_reg
logit=logistic_reg(model_name="lr_titanic",input_relation="train_titanic067",response_column="survived",
                   predictor_columns=["age","gender","family_size","embarked","fare","pclass"],cursor=cur)

# Evaluation of the model importance
logit.features_importance()
```
<p align="center">
<img src='./notebooks/images/logit_fi.png' width="480px">
</p>

```
# Drawing the ROC curve
logit.roc()
```

<p align="center">
<img src='./notebooks/images/titanic_roc.png' width="480px">
</p>

Main advantages:
 - easy data exploration of large dataset using Vertica.
 - easy methods which avoids the call to a huge sql pipeline.
 - easy ML model creation and evaluation.
 - simplify the new functions creation which are hard to create using only sql.

Disadvantages:
 - Vertica-ML-Python will never replace sql and it will never be as fast as using direct sql (direct vsql for example) as some optimizations can not be generated. It is not as complete as sql but it helps to complete it where sql fails.

&#9888; Please read the Vertica ML Python Documentation. If you do not have time just read below.

It is a prototype version (0.1) and it is thanks to all the feedbacks that it can really be improved. 

&#9888; Some of the functions will drastically change in the next release!

If you have any feedback about the library, please contact me: <a href="mailto:badr.ouali@microfocus.com">badr.ouali@microfocus.com</a>

## Prerequires:

Vertica ML Python library is only using the standard Python libraries such as pyodbc, jaydebeapi, matplotlib, time, shutil (only for Python3) and numpy.

## Installation:

Vertica ML Python doesn’t really need installation.
To import easily the Vertica ML Python library from anywhere in your computer just copy paste the entire vertica_ml_python folder in the site-package folder of the Python framework. In the MAC environment, you can find it in: 
```
 /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages
```

## Easy Start:

If you have a DSN and pyodbc is already installed in your machine, write the following command.

```
from vertica_ml_python import RVD
myRVD = RVD('input_relation', dsn='VerticaDSN')
```

You can then see the documentation for the different methods or just enjoy the different tutorials (see the 'notebooks' folder)! The titanic and iris tutorials are perfect to understand the library.
