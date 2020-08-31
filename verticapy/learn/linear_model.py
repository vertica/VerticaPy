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
from verticapy import vDataFrame
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.connections.connect import read_auto_connect
from verticapy.errors import *
from verticapy.learn.vmodel import *

# ---#
class ElasticNet:
    """
---------------------------------------------------------------------------
Creates a ElasticNet object by using the Vertica Highly Distributed and 
Scalable Linear Regression on the data. The Elastic Net is a regularized 
regression method that linearly combines the L1 and L2 penalties of the 
Lasso and Ridge methods. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
penalty: str, optional
	Determines the method of regularization.
		None : No Regularization
		L1   : L1 Regularization
		L2   : L2 Regularization
		ENet : Combination between L1 and L2
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
C: float, optional
	The regularization parameter value. The value must be zero or non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
l1_ratio: float, optional
	ENet mixture parameter that defines how much L1 versus L2 regularization 
	to provide. 

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

coef: tablesample
	Coefficients and their mathematical information (pvalue, std, value...)
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
        penalty: str = "ENet",
        tol: float = 1e-4,
        C: float = 1.0,
        max_iter: int = 100,
        solver: str = "CGD",
        l1_ratio: float = 0.5,
    ):
        check_types(
            [
                ("name", name, [str], False),
                ("solver", solver, ["newton", "bfgs", "cgd"], True),
                ("tol", tol, [int, float], False),
                ("C", C, [int, float], False),
                ("max_iter", max_iter, [int, float], False),
                ("penalty", penalty, ["enet", "l1", "l2", "none"], True),
                ("l1_ratio", l1_ratio, [int, float], False),
            ]
        )
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.type, self.category = "LinearRegression", "regressor"
        self.cursor, self.name = cursor, name
        self.parameters = {
            "penalty": penalty.lower(),
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": solver.lower(),
            "l1_ratio": l1_ratio,
        }

    # ---#
    __repr__ = get_model_repr
    deploySQL = deploySQL
    drop = drop
    features_importance = features_importance
    fit = fit
    get_params = get_params
    plot = plot_model
    predict = predict
    regression_report = regression_metrics_report
    score = regression_score
    set_params = set_params


# ---#
def Lasso(
    name: str, cursor=None, tol: float = 1e-4, max_iter: int = 100, solver: str = "CGD"
):
    """
---------------------------------------------------------------------------
Creates a Lasso object by using the Vertica Highly Distributed and Scalable 
Linear Regression on the data. The Lasso is a regularized regression method 
which uses L1 penalty. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
	"""
    return ElasticNet(
        name=name,
        cursor=cursor,
        penalty="L1",
        tol=tol,
        max_iter=max_iter,
        solver=solver,
    )


# ---#
def LinearRegression(
    name: str,
    cursor=None,
    tol: float = 1e-4,
    max_iter: int = 100,
    solver: str = "Newton",
):
    """
---------------------------------------------------------------------------
Creates a LinearRegression object by using the Vertica Highly Distributed and 
Scalable Linear Regression on the data. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

coef: tablesample
	Coefficients and their mathematical information (pvalue, std, value...)
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
    return ElasticNet(
        name=name,
        cursor=cursor,
        penalty="None",
        tol=tol,
        max_iter=max_iter,
        solver=solver,
    )


# ---#
class LogisticRegression:
    """
---------------------------------------------------------------------------
Creates a LogisticRegression object by using the Vertica Highly Distributed 
and Scalable Logistic Regression on the data.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
penalty: str, optional
	Determines the method of regularization.
		None : No Regularization
		L1   : L1 Regularization
		L2   : L2 Regularization
		ENet : Combination between L1 and L2
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
C: float, optional
	The regularization parameter value. The value must be zero or non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
l1_ratio: float, optional
	ENet mixture parameter that defines how much L1 versus L2 regularization 
	to provide. 

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

coef: tablesample
	Coefficients and their mathematical information (pvalue, std, value...)
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
        penalty: str = "L2",
        tol: float = 1e-4,
        C: int = 1,
        max_iter: int = 100,
        solver: str = "CGD",
        l1_ratio: float = 0.5,
    ):
        check_types(
            [
                ("name", name, [str], False),
                ("solver", solver, ["newton", "bfgs", "cgd"], True),
                ("tol", tol, [int, float], False),
                ("C", C, [int, float], False),
                ("max_iter", max_iter, [int, float], False),
                ("penalty", penalty, ["enet", "l1", "l2", "none"], True),
                ("l1_ratio", l1_ratio, [int, float], False),
            ]
        )
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.type, self.category, self.classes = (
            "LogisticRegression",
            "classifier",
            [0, 1],
        )
        self.cursor, self.name = cursor, name
        self.parameters = {
            "penalty": penalty.lower(),
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": solver.lower(),
            "l1_ratio": l1_ratio,
        }

    # ---#
    __repr__ = get_model_repr
    classification_report = classification_report_binary
    confusion_matrix = confusion_matrix_binary
    deploySQL = deploySQL_binary
    drop = drop
    features_importance = features_importance
    fit = fit
    get_params = get_params
    lift_chart = lift_chart_binary
    plot = plot_model
    prc_curve = prc_curve_binary
    predict = predict_binary
    roc_curve = roc_curve_binary
    score = binary_classification_score
    set_params = set_params


# ---#
def Ridge(
    name: str,
    cursor=None,
    tol: float = 1e-4,
    max_iter: int = 100,
    solver: str = "Newton",
):
    """
---------------------------------------------------------------------------
Creates a Ridge object by using the Vertica Highly Distributed and Scalable 
Linear Regression on the data. The Ridge is a regularized regression method 
which uses L2 penalty. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
	"""
    return ElasticNet(
        name=name,
        cursor=cursor,
        penalty="L2",
        tol=tol,
        max_iter=max_iter,
        solver=solver,
    )
