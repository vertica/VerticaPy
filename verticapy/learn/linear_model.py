# (c) Copyright [2018-2022] Micro Focus or one of its affiliates.
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
# VerticaPy is a Python library with scikit-like functionality for conducting
# data science projects on data stored in Vertica, taking advantage Vertica’s
# speed and built-in analytics and machine learning features. It supports the
# entire data science life cycle, uses a ‘pipeline’ mechanism to sequentialize
# data transformation operations, and offers beautiful graphical options.
#
# VerticaPy aims to do all of the above. The idea is simple: instead of moving
# data around for processing, VerticaPy brings the logic to the data.
#
#
# Modules
#
# VerticaPy Modules
from verticapy import vDataFrame
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *
from verticapy.learn.vmodel import *

# ---#
class ElasticNet(Regressor):
    """
---------------------------------------------------------------------------
Creates a ElasticNet object using the Vertica Linear Regression algorithm 
on the data. The Elastic Net is a regularized regression method that 
linearly combines the L1 and L2 penalties of the Lasso and Ridge methods.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy 
    result.
C: float, optional
	The regularization parameter value. The value must be zero or 
    non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs 
    before achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
l1_ratio: float, optional
	ENet mixture parameter that defines how much L1 versus L2 
    regularization to provide.
	"""

    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        C: float = 1.0,
        max_iter: int = 100,
        solver: str = "CGD",
        l1_ratio: float = 0.5,
    ):
        version(condition=[8, 0, 0])
        check_types([("name", name, [str])])
        self.type, self.name = "LinearRegression", name
        self.set_params(
            {
                "penalty": "enet",
                "tol": tol,
                "C": C,
                "max_iter": max_iter,
                "solver": str(solver).lower(),
                "l1_ratio": l1_ratio,
            }
        )


# ---#
class Lasso(Regressor):
    """
---------------------------------------------------------------------------
Creates a Lasso object using the Vertica Linear Regression algorithm on the 
data. The Lasso is a regularized regression method which uses an L1 penalty.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy 
    result.
C: float, optional
    The regularization parameter value. The value must be zero or 
    non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs 
    before achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
		CGD    : Coordinate Gradient Descent
	"""

    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        C: float = 1.0,
        max_iter: int = 100,
        solver: str = "CGD",
    ):
        version(condition=[8, 0, 0])
        check_types([("name", name, [str])])
        self.type, self.name = "LinearRegression", name
        self.set_params(
            {
                "penalty": "l1",
                "tol": tol,
                "C": C,
                "max_iter": max_iter,
                "solver": str(solver).lower(),
            }
        )
        for elem in ["l1_ratio"]:
            if elem in self.parameters:
                del self.parameters[elem]


# ---#
class LinearRegression(Regressor):
    """
---------------------------------------------------------------------------
Creates a LinearRegression object using the Vertica Linear Regression 
algorithm on the data.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy 
    result.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs 
    before achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
	"""

    def __init__(
        self, name: str, tol: float = 1e-6, max_iter: int = 100, solver: str = "Newton"
    ):
        version(condition=[8, 0, 0])
        check_types(
            [("name", name, [str]), ("solver", solver.lower(), ["newton", "bfgs"])]
        )
        self.type, self.name = "LinearRegression", name
        self.set_params(
            {
                "penalty": "none",
                "tol": tol,
                "max_iter": max_iter,
                "solver": str(solver).lower(),
            }
        )
        for elem in ["l1_ratio", "C"]:
            if elem in self.parameters:
                del self.parameters[elem]


# ---#
class LogisticRegression(BinaryClassifier):
    """
---------------------------------------------------------------------------
Creates a LogisticRegression object using the Vertica Logistic Regression
algorithm on the data.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
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
	"""

    def __init__(
        self,
        name: str,
        penalty: str = "None",
        tol: float = 1e-6,
        C: int = 1,
        max_iter: int = 100,
        solver: str = "Newton",
        l1_ratio: float = 0.5,
    ):
        version(condition=[8, 0, 0])
        check_types([("name", name, [str])])
        self.type, self.name = "LogisticRegression", name
        self.set_params(
            {
                "penalty": str(penalty).lower(),
                "tol": tol,
                "C": C,
                "max_iter": max_iter,
                "solver": str(solver).lower(),
                "l1_ratio": l1_ratio,
            }
        )
        if penalty.lower() == "none":
            for elem in ["l1_ratio", "C"]:
                if elem in self.parameters:
                    del self.parameters[elem]
            check_types([("solver", solver.lower(), ["bfgs", "newton"])])
        elif penalty.lower() in ("l1", "l2"):
            for elem in ["l1_ratio"]:
                if elem in self.parameters:
                    del self.parameters[elem]
            check_types([("solver", solver.lower(), ["bfgs", "newton", "cgd"])])


# ---#
class Ridge(Regressor):
    """
---------------------------------------------------------------------------
Creates a Ridge object using the Vertica Linear Regression algorithm on the 
data. The Ridge is a regularized regression method which uses an L2 penalty. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
tol: float, optional
	Determines whether the algorithm has reached the specified 
    accuracy result.
C: float, optional
    The regularization parameter value. The value must be zero 
    or non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm 
    performs before achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		Newton : Newton Method
		BFGS   : Broyden Fletcher Goldfarb Shanno
	"""

    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        C: float = 1.0,
        max_iter: int = 100,
        solver: str = "Newton",
    ):
        version(condition=[8, 0, 0])
        check_types(
            [("name", name, [str], ("solver", solver.lower(), ["newton", "bfgs"]))]
        )
        self.type, self.name = "LinearRegression", name
        self.set_params(
            {
                "penalty": "l2",
                "tol": tol,
                "C": C,
                "max_iter": max_iter,
                "solver": str(solver).lower(),
            }
        )
        for elem in ["l1_ratio"]:
            if elem in self.parameters:
                del self.parameters[elem]
