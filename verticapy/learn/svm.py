# (c) Copyright [2018-2021] Micro Focus or one of its affiliates.
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
from verticapy.learn.metrics import *
from verticapy.learn.mlplot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.errors import *
from verticapy.learn.vmodel import *

# ---#
class LinearSVC(BinaryClassifier):
    """
---------------------------------------------------------------------------
Creates a LinearSVC object using the Vertica Support Vector Machine (SVM) 
algorithm on the data. Given a set of training examples, each marked as 
belonging to one or the other of two categories, an SVM training algorithm 
builds a model that assigns new examples to one category or the other, making 
it a non-probabilistic binary linear classifier.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica database cursor.
tol: float, optional
	to use to control accuracy.
C: float, optional
	The weight for misclassification cost. The algorithm minimizes the 
	regularization cost and the misclassification cost.
fit_intercept: bool, optional
	A bool to fit also the intercept.
intercept_scaling: float
	A float value, serves as the value of a dummy feature whose coefficient 
	Vertica uses to calculate the model intercept. Because the dummy feature 
	is not in the training data, its values are set to a constant, by default 
	set to 1. 
intercept_mode: str, optional
	Specify how to treat the intercept.
		regularized   : Fits the intercept and applies a regularization on it.
		unregularized : Fits the intercept but does not include it in regularization. 
class_weight: list, optional
	Specifies how to determine weights of the two classes. It can be a list of 2 
	elements or one of the following method:
		auto : Weights each class according to the number of samples.
		none : No weights are used.
max_iter: int, optional
	The maximum number of iterations that the algorithm performs.
	"""

    def __init__(
        self,
        name: str,
        cursor=None,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        intercept_mode: str = "regularized",
        class_weight: list = [1, 1],
        max_iter: int = 100,
    ):
        check_types([("name", name, [str],)])
        self.type, self.name = "LinearSVC", name
        self.set_params(
            {
                "tol": tol,
                "C": C,
                "fit_intercept": fit_intercept,
                "intercept_scaling": intercept_scaling,
                "intercept_mode": intercept_mode,
                "class_weight": class_weight,
                "max_iter": max_iter,
            }
        )
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor
        version(cursor=cursor, condition=[8, 1, 0])


# ---#
class LinearSVR(Regressor):
    """
---------------------------------------------------------------------------
Creates a LinearSVR object using the Vertica SVM (Support Vector Machine) 
algorithm. This algorithm finds the hyperplane used to approximate 
distribution of the data..

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica database cursor.
tol: float, optional
	To use to control accuracy.
C: float, optional
	The weight for misclassification cost. The algorithm minimizes the 
	regularization cost and the misclassification cost.
fit_intercept: bool, optional
	A bool to fit also the intercept.
intercept_scaling: float
	A float value, serves as the value of a dummy feature whose coefficient 
	Vertica uses to calculate the model intercept. Because the dummy feature 
	is not in the training data, its values are set to a constant, by default 
	set to 1. 
intercept_mode: str, optional
	Specify how to treat the intercept.
		regularized   : Fits the intercept and applies a regularization on it.
		unregularized : Fits the intercept but does not include it in regularization. 
acceptable_error_margin: float, optional
	Defines the acceptable error margin. Any data points outside this region add a 
	penalty to the cost function. 
max_iter: int, optional
	The maximum number of iterations that the algorithm performs.

Attributes
----------
After the object creation, all the parameters become attributes. 
The model will also create extra attributes when fitting the model:

coef: tablesample
	Coefficients and their mathematical information (pvalue, std, value...)
input_relation: str
	Training relation.
X: list
	List of the predictors.
y: str
	Response column.
test_relation: str
	Relation used to test the model. All the model methods are abstractions
	which will simplify the process. The test relation will be used by many
	methods to evaluate the model. If empty, the training relation will be
	used as test. You can change it anytime by changing the test_relation
	attribute of the object.
	"""

    def __init__(
        self,
        name: str,
        cursor=None,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        intercept_mode: str = "regularized",
        acceptable_error_margin: float = 0.1,
        max_iter: int = 100,
    ):
        check_types([("name", name, [str],)])
        self.type, self.name = "LinearSVR", name
        self.set_params(
            {
                "tol": tol,
                "C": C,
                "fit_intercept": fit_intercept,
                "intercept_scaling": intercept_scaling,
                "intercept_mode": intercept_mode,
                "acceptable_error_margin": acceptable_error_margin,
                "max_iter": max_iter,
            }
        )
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor
        version(cursor=cursor, condition=[8, 1, 1])
