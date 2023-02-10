"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""

#
#
# Modules
#
# VerticaPy Modules
from verticapy import vDataFrame
from verticapy.utils._decorators import (
    save_verticapy_logs,
    check_minimum_version,
)
from verticapy._version import vertica_version
from verticapy.errors import *
from verticapy.learn.vmodel import *

# Standard Module
from typing import Literal


class ElasticNet(Regressor):
    """
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
C: int / float, optional
	The regularization parameter value. The value must be zero or 
    non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs 
    before achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		newton : Newton Method
		bfgs   : Broyden Fletcher Goldfarb Shanno
		cgd    : Coordinate Gradient Descent
l1_ratio: float, optional
	ENet mixture parameter that defines how much L1 versus L2 
    regularization to provide.
fit_intercept: bool, optional
    Boolean, specifies whether the model includes an intercept. 
    By setting to false, no intercept will be used in training the model. 
    Note that setting fit_intercept to false does not work well with the 
    BFGS optimizer.
	"""

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        C: Union[int, float] = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs", "cgd"] = "cgd",
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
    ):
        self.type, self.name = "LinearRegression", name
        self.VERTICA_FIT_FUNCTION_SQL = "LINEAR_REG"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_LINEAR_REG"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "REGRESSOR"
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise ParameterError(
                "The parameter fit_intercept is only available for Vertica "
                "versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "enet",
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "l1_ratio": l1_ratio,
            "fit_intercept": fit_intercept,
        }


class Lasso(Regressor):
    """
Creates a Lasso object using the Vertica Linear Regression algorithm on the 
data. The Lasso is a regularized regression method which uses an L1 penalty.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy 
    result.
C: int / float, optional
    The regularization parameter value. The value must be zero or 
    non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs 
    before achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		newton : Newton Method
		bfgs   : Broyden Fletcher Goldfarb Shanno
		cgd    : Coordinate Gradient Descent
fit_intercept: bool, optional
    Boolean, specifies whether the model includes an intercept. 
    By setting to false, no intercept will be used in training the model. 
    Note that setting fit_intercept to false does not work well with the 
    BFGS optimizer.
	"""

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        C: Union[int, float] = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs", "cgd"] = "cgd",
        fit_intercept: bool = True,
    ):
        self.type, self.name = "LinearRegression", name
        self.VERTICA_FIT_FUNCTION_SQL = "LINEAR_REG"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_LINEAR_REG"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "REGRESSOR"
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise ParameterError(
                "The parameter fit_intercept is only available for Vertica "
                "versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "l1",
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "fit_intercept": fit_intercept,
        }


class LinearRegression(Regressor):
    """
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
		newton : Newton Method
		bfgs   : Broyden Fletcher Goldfarb Shanno
fit_intercept: bool, optional
    Boolean, specifies whether the model includes an intercept. 
    By setting to false, no intercept will be used in training the model. 
    Note that setting fit_intercept to false does not work well with the 
    BFGS optimizer.
	"""

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs"] = "newton",
        fit_intercept: bool = True,
    ):
        self.type, self.name = "LinearRegression", name
        self.VERTICA_FIT_FUNCTION_SQL = "LINEAR_REG"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_LINEAR_REG"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "REGRESSOR"
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise ParameterError(
                "The parameter fit_intercept is only available for Vertica "
                "versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "none",
            "tol": tol,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "fit_intercept": fit_intercept,
        }


class LogisticRegression(BinaryClassifier):
    """
Creates a LogisticRegression object using the Vertica Logistic Regression
algorithm on the data.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
penalty: str, optional
	Determines the method of regularization.
		None : No Regularization
		l1   : L1 Regularization
		l2   : L2 Regularization
		enet : Combination between L1 and L2
tol: float, optional
	Determines whether the algorithm has reached the specified accuracy result.
C: int / float, optional
	The regularization parameter value. The value must be zero or non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm performs before 
	achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		newton : Newton Method
		bfgs   : Broyden Fletcher Goldfarb Shanno
		cgd    : Coordinate Gradient Descent
l1_ratio: float, optional
	ENet mixture parameter that defines how much L1 versus L2 regularization 
	to provide.
fit_intercept: bool, optional
    Boolean, specifies whether the model includes an intercept. 
    By setting to false, no intercept will be used in training the model. 
    Note that setting fit_intercept to false does not work well with the 
    BFGS optimizer.
	"""

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        penalty: Literal["none", "l1", "l2", "enet", None] = "none",
        tol: float = 1e-6,
        C: Union[int, float] = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs", "cgd"] = "newton",
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
    ):
        penalty = str(penalty).lower()
        solver = str(solver).lower()
        self.type, self.name = "LogisticRegression", name
        self.VERTICA_FIT_FUNCTION_SQL = "LOGISTIC_REG"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_LOGISTIC_REG"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "CLASSIFIER"
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise ParameterError(
                "The parameter fit_intercept is only available for Vertica "
                "versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": penalty,
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": solver,
            "l1_ratio": l1_ratio,
            "fit_intercept": fit_intercept,
        }
        if str(penalty).lower() == "none":
            del self.parameters["l1_ratio"]
            del self.parameters["C"]
            if "solver" == "cgd":
                raise ValueError(
                    "solver can not be set to 'cgd' when there is no regularization."
                )
        elif str(penalty).lower() in ("l1", "l2"):
            del self.parameters["l1_ratio"]


class Ridge(Regressor):
    """
Creates a Ridge object using the Vertica Linear Regression algorithm on the 
data. The Ridge is a regularized regression method which uses an L2 penalty. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
tol: float, optional
	Determines whether the algorithm has reached the specified 
    accuracy result.
C: int / float, optional
    The regularization parameter value. The value must be zero 
    or non-negative.
max_iter: int, optional
	Determines the maximum number of iterations the algorithm 
    performs before achieving the specified accuracy result.
solver: str, optional
	The optimizer method to use to train the model. 
		newton : Newton Method
		bfgs   : Broyden Fletcher Goldfarb Shanno
fit_intercept: bool, optional
    Boolean, specifies whether the model includes an intercept. 
    By setting to false, no intercept will be used in training the model. 
    Note that setting fit_intercept to false does not work well with the 
    BFGS optimizer.
	"""

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str,
        tol: float = 1e-6,
        C: Union[int, float] = 1.0,
        max_iter: int = 100,
        solver: Literal["newton", "bfgs"] = "newton",
        fit_intercept: bool = True,
    ):
        self.type, self.name = "LinearRegression", name
        self.VERTICA_FIT_FUNCTION_SQL = "LINEAR_REG"
        self.VERTICA_PREDICT_FUNCTION_SQL = "PREDICT_LINEAR_REG"
        self.MODEL_TYPE = "SUPERVISED"
        self.MODEL_SUBTYPE = "REGRESSOR"
        if vertica_version()[0] < 12 and not (fit_intercept):
            raise ParameterError(
                "The parameter fit_intercept is only available for Vertica "
                "versions greater or equal to 12."
            )
        self.parameters = {
            "penalty": "l2",
            "tol": tol,
            "C": C,
            "max_iter": max_iter,
            "solver": str(solver).lower(),
            "fit_intercept": fit_intercept,
        }
