# (c) Copyright [2018] Micro Focus or one of its affiliates. 
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AUTHOR: BADR OUALI
#
############################################################################################################ 
#  __ __   ___ ____  ______ ____   __  ____      ___ ___ _          ____  __ __ ______ __ __  ___  ____    #
# |  |  | /  _|    \|      |    | /  ]/    |    |   |   | |        |    \|  |  |      |  |  |/   \|    \   #
# |  |  |/  [_|  D  |      ||  | /  /|  o  |    | _   _ | |        |  o  |  |  |      |  |  |     |  _  |  #
# |  |  |    _|    /|_|  |_||  |/  / |     |    |  \_/  | |___     |   _/|  ~  |_|  |_|  _  |  O  |  |  |  #
# |  :  |   [_|    \  |  |  |  /   \_|  _  |    |   |   |     |    |  |  |___, | |  | |  |  |     |  |  |  #
#  \   /|     |  .  \ |  |  |  \     |  |  |    |   |   |     |    |  |  |     | |  | |  |  |     |  |  |  #
#   \_/ |_____|__|\_| |__| |____\____|__|__|    |___|___|_____|    |__|  |____/  |__| |__|__|\___/|__|__|  #
#                                                                                                          #
############################################################################################################
# Vertica-ML-Python allows user to create Virtual Dataframe. vDataframes simplify   #
# data exploration,   data cleaning   and   machine   learning   in    Vertica.     #
# It is an object which keeps in it all the actions that the user wants to achieve  # 
# and execute them when they are needed.    										#
#																					#
# The purpose is to bring the logic to the data and not the opposite                #
#####################################################################################
#
import setuptools

setuptools.setup(
	name = 'vertica_ml_python',  
    version = '1.0',
	scripts = ['vertica_ml_python'],
    author = "Badr Ouali",
	author_email = "badr.ouali@microfocus.com",
	description = "Vertica-ML-Python simplifies data exploration, data cleaning and machine learning in Vertica.",
	long_description = "Vertica-ML-Python allows users to use the vDataframe (Virtual Dataframe). This object keeps in memory all the users modifications in order to use optimised SQL queries to compute all the necessary aggregations. Thanks to this object, the table is intact and will never be modified. The purpose is to explore, preprocess and clean the object without changing the initial table.\nThis library contains many functions for:\n\t• Data Exploration, Preprocessing and Cleaning: vertica_ml_python.vdataframe\n\t• Machine Learning (Regression, Classification, Clustering): vertica_ml_python.learn\nvertica-ml-python helps to explore, preprocess and clean the data without changing the initial table. It uses scalable Machine Learning Algorithms such as Logistic Regression, Random Forest and SVM. It allows also to evaluate and to optimise models (Classification/Regression Reports, ROC/PRC curves, Parameters tuning...).\nvertica-ml-python uses only the standard Python libraries. To connect to the database, it can use both JDBC and ODBC connection. Everything the user needs is a DSN or a Database cursor (having both: ’execute’ and ’fetchall’ methods). Significant examples using very well-known datasets are available and will help the user to master the different objects.",
	long_description_content_type="text/markdown",
	url = "https://github.com/vertica/Vertica-ML-Python",
	packages = setuptools.find_packages(),
	classifiers=["Programming Language :: Python :: 3", "License :: Apache License, Version 2.0", "Operating System :: OS Independent",],)

