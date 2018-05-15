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
# Vertica-ML-Python allows user to create  RVD (Resilient Vertica Dataset).         #
# RVD  simplifies data exploration, data cleaning and machine learning in  Vertica. #
# It is an object which keeps in it all the actions that the user wants to achieve  # 
# and execute them when they are needed.                                            #
#####################################################################################
#                    #
# Author: Badr Ouali #
#                    #
######################

"""
Vertica-ML-Python is a Python library that exposes sci-kit like functionality to conduct 
data science projects on data stored in Vertica, thus taking advantage Vertica’s speed and 
built-in analytics and machine learning capabilities. It supports the entire data science 
life cycle, uses a ‘pipeline’ mechanism to sequentialize data transformation operation 
(called Resilient Vertica Dataset), and offers multiple graphical rendering possibilities.

See:
https://github.com/vertica/vertica_ml_python
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

setup(
    name='vertica_ml_python',
    version='0.1',
    description='Vertica-ML-Python simplifies data exploration, data cleaning and machine learning in Vertica.',
    long_description='Vertica-ML-Python is a Python library that exposes sci-kit like functionality to conduct data science projects on data stored in Vertica, thus taking advantage Vertica’s speed and built-in analytics and machine learning capabilities. It supports the entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize data transformation operation (called Resilient Vertica Dataset), and offers multiple graphical rendering possibilities.',  
    long_description_content_type='text/markdown',
    url='https://github.com/vertica/vertica_ml_python',
    author='Badr Ouali',
    author_email='badr.ouali@microfocus.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
    ],
    keywords='vertica machine-learning python big-data data-visualization preparation data-science python-library',
    packages=find_packages(),
    package_dir={'':'vertica_ml_python'},
    install_requires=['numpy','math','matplotlib','random','time','shutil'], 
    project_urls={ 
        'Bug Reports': 'https://github.com/vertica/vertica_ml_python/issues',
        'Say Thanks!': 'badr.ouali@microfocus.com',
        'Source': 'https://github.com/vertica/vertica_ml_python',
    },
)
