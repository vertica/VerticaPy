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
from verticapy.learn.metrics import *
from verticapy.learn.mlplot import *
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy import vDataFrame
from verticapy.errors import *
from verticapy.learn.vmodel import *

# ---#
class NaiveBayes(MulticlassClassifier):
    """
---------------------------------------------------------------------------
Creates a NaiveBayes object using the Vertica Naive Bayes algorithm on 
the data. It is a "probabilistic classifier" based on applying Bayes' theorem 
with strong (naïve) independence assumptions between the features.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
cursor: DBcursor, optional
	Vertica database cursor. 
alpha: float, optional
	A float that specifies use of Laplace smoothing if the event model is 
	categorical, multinomial, or Bernoulli.
nbtype: str, optional
    Naive Bayes Type.
    - auto        : Vertica NB will treat columns according to data type:
        - FLOAT        : Values are assumed to follow some Gaussian 
                         distribution.
        - INTEGER      : Values are assumed to belong to one multinomial 
                         distribution.
        - CHAR/VARCHAR : Values are assumed to follow some categorical 
                         distribution. The string values stored in these 
                         columns must not be greater than 128 characters.
        - BOOLEAN      : Values are treated as categorical with two values.
     - bernoulli   : Casts the variables to boolean.
     - categorical : Casts the variables to categorical.
     - multinomial : Casts the variables to integer.
     - gaussian    : Casts the variables to float.
	"""

    def __init__(
        self, name: str, cursor=None, alpha: float = 1.0, nbtype: str = "auto"
    ):
        check_types(
            [
                ("name", name, [str],),
                ("alpha", alpha, [int, float],),
                (
                    "nbtype",
                    nbtype,
                    ["auto", "bernoulli", "categorical", "multinomial", "gaussian"],
                ),
            ]
        )
        self.type, self.name = "NaiveBayes", name
        self.set_params({"alpha": alpha, "nbtype": nbtype})
        cursor = check_cursor(cursor)[0]
        self.cursor = cursor
        version(cursor=cursor, condition=[8, 0, 0])


# ---#
class BernoulliNB(NaiveBayes):
    """i.e. NaiveBayes with param nbtype = 'bernoulli'"""

    def __init__(
        self, name: str, cursor=None, alpha: float = 1.0,
    ):
        super().__init__(name, cursor, alpha, "bernoulli")


# ---#
class CategoricalNB(NaiveBayes):
    """i.e. NaiveBayes with param nbtype = 'categorical'"""

    def __init__(
        self, name: str, cursor=None, alpha: float = 1.0,
    ):
        super().__init__(name, cursor, alpha, "categorical")


# ---#
class GaussianNB(NaiveBayes):
    """i.e. NaiveBayes with param nbtype = 'gaussian'"""

    def __init__(
        self, name: str, cursor=None,
    ):
        super().__init__(name, cursor, nbtype="gaussian")


# ---#
class MultinomialNB(NaiveBayes):
    """i.e. NaiveBayes with param nbtype = 'multinomial'"""

    def __init__(
        self, name: str, cursor=None, alpha: float = 1.0,
    ):
        super().__init__(name, cursor, alpha, "multinomial")
