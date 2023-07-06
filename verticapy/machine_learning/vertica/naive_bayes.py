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
from typing import Literal
import numpy as np

from verticapy._typing import PythonNumber
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import quote_ident
from verticapy._utils._sql._vertica_version import check_minimum_version

from verticapy.core.vdataframe.base import vDataFrame

import verticapy.machine_learning.memmodel as mm

from verticapy.machine_learning.vertica.base import MulticlassClassifier

"""
Algorithms used for classification.
"""


class NaiveBayes(MulticlassClassifier):
    """
    Creates  a  NaiveBayes object using the Vertica
    Naive  Bayes  algorithm.  It is a "probabilistic
    classifier"  based  on  applying Bayes' theorem
    with strong (naïve) independence assumptions
    between the features.

    Parameters
    ----------
    name: str, optional
        Name  of  the  model.  The  model is stored
        in the database.
    overwrite_model: bool, optional
        If set to True, training a model with the same
        name as an existing model overwrites the
        existing model.
    alpha: float, optional
        A  float  that  specifies  use  of  Laplace
        smoothing if the event model is categorical,
        multinomial, or Bernoulli.
    nbtype: str, optional
        Naive Bayes type.
        - auto        : Vertica NaiveBayes objects
                treat columns according to data type:
              * FLOAT values are assumed to follow some
                Gaussian distribution.
              * INTEGER values are assumed to belong to
                one multinomial distribution.
              * CHAR/VARCHAR   values  are  assumed  to
                follow  some categorical distribution.
                The  string  values  stored  in  these
                columns  must be no greater than  128
                characters.
              * BOOLEAN    values    are   treated   as
                categorical with two values.
        - bernoulli   : Casts the variables to boolean.
        - categorical : Casts the variables to categorical.
        - multinomial : Casts the variables to integer.
        - gaussian    : Casts the variables to float.
    """

    # Properties.

    @property
    def _vertica_fit_sql(self) -> Literal["NAIVE_BAYES"]:
        return "NAIVE_BAYES"

    @property
    def _vertica_predict_sql(self) -> Literal["PREDICT_NAIVE_BAYES"]:
        return "PREDICT_NAIVE_BAYES"

    @property
    def _model_subcategory(self) -> Literal["CLASSIFIER"]:
        return "CLASSIFIER"

    @property
    def _model_type(self) -> Literal["NaiveBayes"]:
        return "NaiveBayes"

    @property
    def _attributes(self) -> list[str]:
        return ["attributes_", "prior_", "classes_"]

    # System & Special Methods.

    @check_minimum_version
    @save_verticapy_logs
    def __init__(
        self,
        name: str = None,
        overwrite_model: bool = False,
        alpha: PythonNumber = 1.0,
        nbtype: Literal[
            "auto", "bernoulli", "categorical", "multinomial", "gaussian"
        ] = "auto",
    ) -> None:
        super().__init__(name, overwrite_model)
        self.parameters = {"alpha": alpha, "nbtype": str(nbtype).lower()}

    # Attributes Methods.

    def _compute_attributes(self) -> None:
        """
        Computes the model's attributes.
        """
        self.classes_ = self._array_to_int(
            np.array(self.get_vertica_attributes("prior")["class"])
        )
        self.prior_ = np.array(self.get_vertica_attributes("prior")["probability"])
        self.attributes_ = self._get_nb_attributes()

    def _get_nb_attributes(self) -> list[dict]:
        """
        Returns a list of dictionary for each of the NB
        variables. It is used to translate NB to Python.
        """
        vdf = vDataFrame(self.input_relation)
        var_info = {}
        gaussian_incr, bernoulli_incr, multinomial_incr = 0, 0, 0
        for idx, elem in enumerate(self.X):
            var_info[elem] = {"rank": idx}
            if vdf[elem].isbool():
                var_info[elem]["type"] = "bernoulli"
                for c in self.classes_:
                    var_info[elem][c] = self.get_vertica_attributes(f"bernoulli.{c}")[
                        "probability"
                    ][bernoulli_incr]
                bernoulli_incr += 1
            elif vdf[elem].category() == "int":
                var_info[elem]["type"] = "multinomial"
                for c in self.classes_:
                    multinomial = self.get_vertica_attributes(f"multinomial.{c}")
                    var_info[elem][c] = multinomial["probability"][multinomial_incr]
                multinomial_incr += 1
            elif vdf[elem].isnum():
                var_info[elem]["type"] = "gaussian"
                for c in self.classes_:
                    gaussian = self.get_vertica_attributes(f"gaussian.{c}")
                    var_info[elem][c] = {
                        "mu": gaussian["mu"][gaussian_incr],
                        "sigma_sq": gaussian["sigma_sq"][gaussian_incr],
                    }
                gaussian_incr += 1
            else:
                var_info[elem]["type"] = "categorical"
                my_cat = "categorical." + quote_ident(elem)[1:-1]
                attr = self.get_vertica_attributes()["attr_name"]
                for item in attr:
                    if item.lower() == my_cat.lower():
                        my_cat = item
                        break
                val = self.get_vertica_attributes(my_cat).values
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

    # Parameters Methods.

    @staticmethod
    def _map_to_vertica_param_dict() -> dict:
        return {}

    # I/O Methods.

    def to_memmodel(self) -> mm.NaiveBayes:
        """
        Converts  the model to an InMemory object  that
        can be used for different types of predictions.
        """
        return mm.NaiveBayes(
            self.attributes_,
            self.prior_,
            self.classes_,
        )


class BernoulliNB(NaiveBayes):
    """NaiveBayes with parameter nbtype = 'bernoulli'"""

    def __init__(
        self, name: str = None, overwrite_model: bool = False, alpha: float = 1.0
    ) -> None:
        super().__init__(name, overwrite_model, alpha, nbtype="bernoulli")


class CategoricalNB(NaiveBayes):
    """NaiveBayes with parameter nbtype = 'categorical'"""

    def __init__(
        self, name: str = None, overwrite_model: bool = False, alpha: float = 1.0
    ) -> None:
        super().__init__(name, overwrite_model, alpha, nbtype="categorical")


class GaussianNB(NaiveBayes):
    """NaiveBayes with parameter nbtype = 'gaussian'"""

    def __init__(self, name: str = None, overwrite_model: bool = False) -> None:
        super().__init__(name, overwrite_model, nbtype="gaussian")


class MultinomialNB(NaiveBayes):
    """NaiveBayes with parameter nbtype = 'multinomial'"""

    def __init__(
        self, name: str = None, overwrite_model: bool = False, alpha: float = 1.0
    ) -> None:
        super().__init__(name, overwrite_model, alpha, nbtype="multinomial")
