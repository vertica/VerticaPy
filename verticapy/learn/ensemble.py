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
# |_     |~) _  _| _  /~\    _ |.
# |_)\/  |_)(_|(_||   \_/|_|(_|||
#    /
#              ____________       ______
#             / __        `\     /     /
#            |  \/         /    /     /
#            |______      /    /     /
#                   |____/    /     /
#          _____________     /     /
#          \           /    /     /
#           \         /    /     /
#            \_______/    /     /
#             ______     /     /
#             \    /    /     /
#              \  /    /     /
#               \/    /     /
#                    /     /
#                   /     /
#                   \    /
#                    \  /
#                     \/
#                    _
# \  / _  __|_. _ _ |_)
#  \/ (/_|  | |(_(_|| \/
#                     /
# VerticaPy is a Python library with scikit-like functionality to use to conduct
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to solve all of these problems. The idea is simple: instead
# of moving data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# VerticaPy Modules
from verticapy.learn.metrics import *
from verticapy.learn.plot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.connections.connect import read_auto_connect
from verticapy.errors import *
from verticapy.learn.vmodel import *

# ---#
class RandomForestClassifier:
    """
---------------------------------------------------------------------------
Creates a RandomForestClassifier object by using the Vertica Highly Distributed 
and Scalable Random Forest on the data. It is one of the ensemble learning 
method for classification that operate by constructing a multitude of decision 
trees at training time and outputting the class that is the mode of the classes.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor. 
n_estimators: int, optional
	The number of trees in the forest, an integer between 0 and 1000, inclusive.
max_features: str, optional
	The number of randomly chosen features from which to pick the best feature 
	to split on a given tree node. It can be an integer or one of the two following
	methods.
		auto : square root of the total number of predictors.
		max  : number of predictors.
max_leaf_nodes: int, optional
	The maximum number of leaf nodes a tree in the forest can have, an integer 
	between 1 and 1e9, inclusive.
sample: float, optional
	The portion of the input data set that is randomly picked for training each tree, 
	a float between 0.0 and 1.0, inclusive. 
max_depth: int, optional
	The maximum depth for growing each tree, an integer between 1 and 100, inclusive.
min_samples_leaf: int, optional
	The minimum number of samples each branch must have after splitting a node, an 
	integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples 
	is discarded. 
min_info_gain: float, optional
	The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
	A split with information gain less than this threshold is discarded.
nbins: int, optional 
	The number of bins to use for continuous features, an integer between 2 and 1000, 
	inclusive.

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

classes: list
	List of all the response classes.
input_relation: str
	Train relation.
X: list
	List of the predictors.
y: str
	Response column.
test_relation: str
	Relation to use to test the model. All the model methods are abstractions
	which will simplify the process. The test relation will be used by many
	methods to evaluate the model. If empty, the train relation will be 
	used as test. You can change it anytime by changing the test_relation
	attribute of the object.
	"""

    #
    # Special Methods
    #
    # ---#
    def __init__(
        self,
        name: str,
        cursor=None,
        n_estimators: int = 10,
        max_features="auto",
        max_leaf_nodes: int = 1e9,
        sample: float = 0.632,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        min_info_gain: float = 0.0,
        nbins: int = 32,
    ):
        check_types(
            [
                ("name", name, [str], False),
                ("n_estimators", n_estimators, [int, float], False),
                ("max_features", max_features, [str, int, float], False),
                ("max_leaf_nodes", max_leaf_nodes, [int, float], False),
                ("sample", sample, [int, float], False),
                ("max_depth", max_depth, [int, float], False),
                ("min_samples_leaf", min_samples_leaf, [int, float], False),
                ("min_info_gain", min_info_gain, [int, float], False),
                ("nbins", nbins, [int, float], False),
            ]
        )
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.type, self.category = "RandomForestClassifier", "classifier"
        self.cursor, self.name = cursor, name
        self.parameters = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "sample": sample,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }

    # ---#
    __repr__ = get_model_repr
    classification_report = classification_report_multiclass
    confusion_matrix = confusion_matrix_multiclass
    deploySQL = deploySQL_multiclass
    drop = drop
    export_graphviz = export_graphviz
    features_importance = features_importance
    fit = fit
    get_params = get_params
    get_tree = get_tree
    lift_chart = lift_chart_multiclass
    plot_tree = plot_rf_tree
    prc_curve = prc_curve_multiclass
    predict = predict_multiclass
    roc_curve = roc_curve_multiclass
    score = multiclass_classification_score
    set_params = set_params


# ---#
class RandomForestRegressor:
    """
---------------------------------------------------------------------------
Creates a RandomForestRegressor object by using the Vertica Highly Distributed 
and Scalable Random Forest on the data. It is one of the ensemble learning 
method for regression that operate by constructing a multitude of decision 
trees at training time and outputting the mean prediction.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor. 
n_estimators: int, optional
	The number of trees in the forest, an integer between 0 and 1000, inclusive.
max_features: str, optional
	The number of randomly chosen features from which to pick the best feature 
	to split on a given tree node. It can be an integer or one of the two following
	methods.
		auto : square root of the total number of predictors.
		max  : number of predictors.
max_leaf_nodes: int, optional
	The maximum number of leaf nodes a tree in the forest can have, an integer 
	between 1 and 1e9, inclusive.
sample: float, optional
	The portion of the input data set that is randomly picked for training each tree, 
	a float between 0.0 and 1.0, inclusive. 
max_depth: int, optional
	The maximum depth for growing each tree, an integer between 1 and 100, inclusive.
min_samples_leaf: int, optional
	The minimum number of samples each branch must have after splitting a node, an 
	integer between 1 and 1e6, inclusive. A split that causes fewer remaining samples 
	is discarded. 
min_info_gain: float, optional
	The minimum threshold for including a split, a float between 0.0 and 1.0, inclusive. 
	A split with information gain less than this threshold is discarded.
nbins: int, optional 
	The number of bins to use for continuous features, an integer between 2 and 1000, 
	inclusive.

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

input_relation: str
	Train relation.
X: list
	List of the predictors.
y: str
	Response column.
test_relation: str
	Relation to use to test the model. All the model methods are abstractions
	which will simplify the process. The test relation will be used by many
	methods to evaluate the model. If empty, the train relation will be 
	used as test. You can change it anytime by changing the test_relation
	attribute of the object.
	"""

    #
    # Special Methods
    #
    # ---#
    def __init__(
        self,
        name: str,
        cursor=None,
        n_estimators: int = 10,
        max_features="auto",
        max_leaf_nodes: int = 1e9,
        sample: float = 0.632,
        max_depth: int = 5,
        min_samples_leaf: int = 1,
        min_info_gain: float = 0.0,
        nbins: int = 32,
    ):
        check_types(
            [
                ("name", name, [str], False),
                ("n_estimators", n_estimators, [int, float], False),
                ("max_features", max_features, [str, int, float], False),
                ("max_leaf_nodes", max_leaf_nodes, [int, float], False),
                ("sample", sample, [int, float], False),
                ("max_depth", max_depth, [int, float], False),
                ("min_samples_leaf", min_samples_leaf, [int, float], False),
                ("min_info_gain", min_info_gain, [int, float], False),
                ("nbins", nbins, [int, float], False),
            ]
        )
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.type, self.category = "RandomForestRegressor", "regressor"
        self.cursor, self.name = cursor, name
        self.parameters = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "sample": sample,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_info_gain": min_info_gain,
            "nbins": nbins,
        }

    # ---#
    __repr__ = get_model_repr
    deploySQL = deploySQL
    drop = drop
    export_graphviz = export_graphviz
    features_importance = features_importance
    fit = fit
    get_params = get_params
    get_tree = get_tree
    plot_tree = plot_rf_tree
    predict = predict
    regression_report = regression_metrics_report
    score = regression_score
    set_params = set_params
