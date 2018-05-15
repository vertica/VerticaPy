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

__version__ = "0.1"
__author__ = "Badr Ouali"
__author_email__ = "badr.ouali@microfocus.com"
__description__ = """Vertica-ML-Python simplifies data exploration, data cleaning and machine learning in Vertica."""
__url__ = "https://github.com/vertica/vertica_ml_python/"

# RVD
from vertica_ml_python.rvd import RVD
from vertica_ml_python.rvd import drop_table
from vertica_ml_python.rvd import drop_view
from vertica_ml_python.rvd import read_csv

# Fun
from vertica_ml_python.fun import column_matrix
from vertica_ml_python.fun import run_query

# VML
#
# functions
from vertica_ml_python.vml import accuracy
from vertica_ml_python.vml import auc
from vertica_ml_python.vml import champion_challenger_binomial
from vertica_ml_python.vml import confusion_matrix
from vertica_ml_python.vml import details
from vertica_ml_python.vml import drop_model
from vertica_ml_python.vml import elbow
from vertica_ml_python.vml import error_rate
from vertica_ml_python.vml import features_importance
from vertica_ml_python.vml import lift_table
from vertica_ml_python.vml import load_model
from vertica_ml_python.vml import logloss
from vertica_ml_python.vml import metric_rf_curve_ntree
from vertica_ml_python.vml import metric_rf_curve_depth
from vertica_ml_python.vml import mse
from vertica_ml_python.vml import parameter_value
from vertica_ml_python.vml import plot_reg
from vertica_ml_python.vml import reg_metrics
from vertica_ml_python.vml import roc
from vertica_ml_python.vml import rsquared
from vertica_ml_python.vml import summarize_model
from vertica_ml_python.vml import tree
#
# algorithms
#
from vertica_ml_python.vml import cross_validate
from vertica_ml_python.vml import kmeans
from vertica_ml_python.vml import linear_reg
from vertica_ml_python.vml import logistic_reg
from vertica_ml_python.vml import naive_bayes
from vertica_ml_python.vml import rf_classifier
from vertica_ml_python.vml import rf_regressor
from vertica_ml_python.vml import svm_classifier
from vertica_ml_python.vml import svm_regressor
