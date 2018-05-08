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

# vertica_ml_python

Vertica-ML-Python allows user to create  RVD (Resilient Vertica Dataset). RVD  simplifies data exploration, data cleaning and machine learning in  Vertica. It is an object which keeps in it all the actions that the user wants to achieve and execute them when they are needed.   

Main advantages:
 - easy data exploration of large dataset using Vertica.
 - easy methods which avoids the call to a huge sql pipeline.
 - easy ML model creation and evaluation.
 - simplify the new functions creation which are hard to create using only sql.

Disadvantages:
 - Vertica-ML-Python will never replace sql and it will never be as fast as using direct sql (direct vsql for example) as some optimizations can not be generated. It is not as complete as sql but it helps to complete it where sql fails.

/!\ Please read the Vertica ML Python Documentation. If you do not have time just read below.

It is a prototype version (0.1) and it is thanks to all the feedbacks that it can really be improved. 
/!\ Some of the functions will drastically change in the next release!

If you have any feedback about the library please contact me: badr.ouali@microfocus.com

Prerequires:

Vertica ML Python library is only using the standard Python libraries such as pyodbc, jaydebeapi, matplotlib, time, shutil (only for Python3) and numpy.

Installation:

Vertica ML Python doesn’t really need installation.
To import easily the Vertica ML Python library from anywhere in your computer just copy paste the entire vertica_ml_python folder in the site-package folder of the Python framework. In the MAC environment, you can find it in: 
 /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages

Easy Start:

If you have a DSN and pyodbc is already installed in your machine, write the following command.

```
from vertica_ml_python import RVD
myRVD = RVD('input_relation', dsn='VerticaDSN')
```

You can then see the documentation for the different methods or just enjoy the different tutorials!
