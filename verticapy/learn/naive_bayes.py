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
from verticapy.learn.vmodel import *

# ---#
class NaiveBayes(MulticlassClassifier):
    """
---------------------------------------------------------------------------
Creates a NaiveBayes object using the Vertica Naive Bayes algorithm on 
the data. It is a "probabilistic classifier" based on applying Bayes' 
theorem with strong (naïve) independence assumptions between the features.

Parameters
----------
name: str
	Name of the the model. The model will be stored in the DB.
alpha: float, optional
	A float that specifies use of Laplace smoothing if the event model is 
	categorical, multinomial, or Bernoulli.
nbtype: str, optional
    Naive Bayes Type.
    - auto        : Vertica NB will treat columns according to data type:
        * FLOAT        : Values are assumed to follow some Gaussian 
                         distribution.
        * INTEGER      : Values are assumed to belong to one multinomial 
                         distribution.
        * CHAR/VARCHAR : Values are assumed to follow some categorical 
                         distribution. The string values stored in these 
                         columns must not be greater than 128 characters.
        * BOOLEAN      : Values are treated as categorical with two values.
     - bernoulli   : Casts the variables to boolean.
     - categorical : Casts the variables to categorical.
     - multinomial : Casts the variables to integer.
     - gaussian    : Casts the variables to float.
	"""

    def __init__(self, name: str, alpha: float = 1.0, nbtype: str = "auto"):

        nbtype_vals = ["auto", "bernoulli", "categorical", "multinomial", "gaussian"]
        check_types(
            [
                ("name", name, [str]),
                ("alpha", alpha, [int, float]),
                ("nbtype", nbtype, nbtype_vals),
            ]
        )
        self.type, self.name = "NaiveBayes", name
        self.set_params({"alpha": alpha, "nbtype": nbtype})
        version(condition=[8, 0, 0])

    # ---#
    def get_var_info(self):
        # Returns a list of dictionary for each of the NB variables.
        # It is used to translate NB to Python
        from verticapy.utilities import vDataFrameSQL

        vdf = vDataFrameSQL(self.input_relation)
        var_info = {}
        gaussian_incr, bernoulli_incr, multinomial_incr = 0, 0, 0
        for idx, elem in enumerate(self.X):
            var_info[elem] = {"rank": idx}
            if vdf[elem].isbool():
                var_info[elem]["type"] = "bernoulli"
                for c in self.classes_:
                    var_info[elem][c] = self.get_attr("bernoulli.{}".format(c))[
                        "probability"
                    ][bernoulli_incr]
                bernoulli_incr += 1
            elif vdf[elem].category() == "int":
                var_info[elem]["type"] = "multinomial"
                for c in self.classes_:
                    multinomial = self.get_attr("multinomial.{}".format(c))
                    var_info[elem][c] = multinomial["probability"][multinomial_incr]
                multinomial_incr += 1
            elif vdf[elem].isnum():
                var_info[elem]["type"] = "gaussian"
                for c in self.classes_:
                    gaussian = self.get_attr("gaussian.{}".format(c))
                    var_info[elem][c] = {
                        "mu": gaussian["mu"][gaussian_incr],
                        "sigma_sq": gaussian["sigma_sq"][gaussian_incr],
                    }
                gaussian_incr += 1
            else:
                var_info[elem]["type"] = "categorical"
                my_cat = "categorical." + quote_ident(elem)[1:-1]
                attr = self.get_attr()["attr_name"]
                for item in attr:
                    if item.lower() == my_cat.lower():
                        my_cat = item
                        break
                val = self.get_attr(my_cat).values
                for item in val:
                    if item != "category":
                        if item not in var_info[elem]:
                            var_info[elem][item] = {}
                        for i, p in enumerate(val[item]):
                            var_info[elem][item][val["category"][i]] = p
        var_info_simplified = []
        for i in range(len(var_info)):
            for elem in var_info:
                if var_info[elem]["rank"] == i:
                    var_info_simplified += [var_info[elem]]
                    break
        for elem in var_info_simplified:
            del elem["rank"]
        return var_info_simplified


# ---#
class BernoulliNB(NaiveBayes):
    """i.e. NaiveBayes with param nbtype = 'bernoulli'"""

    def __init__(self, name: str, alpha: float = 1.0):

        super().__init__(name, alpha, "bernoulli")


# ---#
class CategoricalNB(NaiveBayes):
    """i.e. NaiveBayes with param nbtype = 'categorical'"""

    def __init__(self, name: str, alpha: float = 1.0):

        super().__init__(name, alpha, "categorical")


# ---#
class GaussianNB(NaiveBayes):
    """i.e. NaiveBayes with param nbtype = 'gaussian'"""

    def __init__(self, name: str):

        super().__init__(name, nbtype="gaussian")


# ---#
class MultinomialNB(NaiveBayes):
    """i.e. NaiveBayes with param nbtype = 'multinomial'"""

    def __init__(self, name: str, alpha: float = 1.0):

        super().__init__(name, alpha, "multinomial")
