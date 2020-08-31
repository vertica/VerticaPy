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
class MultinomialNB:
    """
---------------------------------------------------------------------------
Creates a MultinomialNB object by using the Vertica Highly Distributed 
and Scalable Naive Bayes on the data. It is a "probabilistic classifiers" 
based on applying Bayes theorem with strong (naïve) independence assumptions 
between the features. 

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica DB cursor. 
alpha: float, optional
	A float that specifies use of Laplace smoothing if the event model is 
	categorical, multinomial, or Bernoulli.

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
    def __init__(self, name: str, cursor=None, alpha: float = 1.0):
        check_types(
            [("name", name, [str], False), ("alpha", alpha, [int, float], False)]
        )
        if not (cursor):
            cursor = read_auto_connect().cursor()
        else:
            check_cursor(cursor)
        self.type, self.category = "MultinomialNB", "classifier"
        self.cursor, self.name = cursor, name
        self.parameters = {"alpha": alpha}

    # ---#
    __repr__ = get_model_repr
    classification_report = classification_report_multiclass
    confusion_matrix = confusion_matrix_multiclass
    deploySQL = deploySQL_multiclass
    drop = drop
    fit = fit
    get_params = get_params
    lift_chart = lift_chart_multiclass
    prc_curve = prc_curve_multiclass
    predict = predict_multiclass
    roc_curve = roc_curve_multiclass
    score = multiclass_classification_score
    set_params = set_params
