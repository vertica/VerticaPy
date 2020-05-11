# (c) Copyright [2018-2020] Micro Focus or one of its affiliates. 
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

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
	name = 'vertica_ml_python',  
    version = '1.0-beta',
    author = "Badr Ouali",
	author_email = "badr.ouali@microfocus.com",
	url = "https://github.com/vertica/Vertica-ML-Python",
	keywords = "vertica python ml data science machine learning statistics database",
	description = "Vertica-ML-Python simplifies data exploration, data cleaning and machine learning in Vertica.",
	long_description = long_description,
	long_description_content_type = "text/markdown",
	packages = setuptools.find_packages(),
	python_requires = ">=3.6",
	install_requires = [
        'matplotlib>=2.0'
    ],
	package_data = {'': ['*.csv']},
	classifiers = [
		"Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.6", 
		"Programming Language :: Python :: 3.7", 
		"Programming Language :: Python :: 3.8", 
		"Topic :: Database",
		"License :: OSI Approved :: Apache Software License", 
		"Operating System :: OS Independent",],
	)