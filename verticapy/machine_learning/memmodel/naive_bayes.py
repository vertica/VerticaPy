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
import copy
from typing import Literal

import numpy as np

from verticapy._typing import ArrayLike
from verticapy._utils._sql._format import clean_query

from verticapy.machine_learning.memmodel.base import MulticlassClassifier


class NaiveBayes(MulticlassClassifier):
    """
    InMemoryModel implementation of the naive Bayes algorithm.

    Parameters
    ----------
    attributes: list
        List  of the model's attributes. Each feature  must
        be represented by a dictionary, which differs based
        on the distribution.
          For 'gaussian':
            Key 'type'  must have 'gaussian' as value.
            Each of the model's classes must include a
            dictionary with two keys:
              sigma_sq: Square  root of  the  standard
                        deviation.
              mu: Average.
            Example: {'type': 'gaussian',
                      'C': {'mu': 63.9878308300395,
                            'sigma_sq': 7281.87598377196},
                      'Q': {'mu': 13.0217386792453,
                            'sigma_sq': 211.626862330204},
                      'S': {'mu': 27.6928120412844,
                            'sigma_sq': 1428.57067393938}}
          For 'multinomial':
            Key 'type' must have 'multinomial' as value.
            Each of the model's classes must be represented
            by a key with its probability as the value.
            Example: {'type': 'multinomial',
                      'C': 0.771666666666667,
                      'Q': 0.910714285714286,
                      'S': 0.878216123499142}
          For 'bernoulli':
            Key 'type' must have 'bernoulli' as value.
            Each of the model's classes must be represented
            by a key with its probability as the value.
            Example: {'type': 'bernoulli',
                      'C': 0.537254901960784,
                      'Q': 0.277777777777778,
                      'S': 0.324942791762014}
          For 'categorical':
            Key 'type' must have 'categorical' as value.
            Each  of  the  model's  classes  must  include
            a dictionary with all the feature categories.
            Example: {'type': 'categorical',
                      'C': {'female': 0.407843137254902,
                            'male': 0.592156862745098},
                      'Q': {'female': 0.416666666666667,
                            'male': 0.583333333333333},
                      'S': {'female': 0.311212814645309,
                            'male': 0.688787185354691}}
    prior: ArrayLike
        The model's classes probabilities.
    classes: ArrayLike
        The model's classes.
    """

    # Properties.

    @property
    def object_type(self) -> Literal["NaiveBayes"]:
        return "NaiveBayes"

    @property
    def _attributes(self) -> list[str]:
        return ["attributes_", "prior_", "classes_"]

    # System & Special Methods.

    def __init__(
        self,
        attributes: list[dict],
        prior: ArrayLike,
        classes: ArrayLike,
    ) -> None:
        self.attributes_ = copy.deepcopy(attributes)
        self.prior_ = np.array(prior)
        self.classes_ = np.array(classes)

    # Prediction / Transformation Methods - IN MEMORY.

    def _predict_row(self, X: ArrayLike, return_proba: bool = False) -> np.ndarray:
        """
        Predicts for one row.
        """
        res = []
        for c in self.classes_:
            sub_result = []
            for idx in range(len(X)):
                prob = self.attributes_[idx]
                if prob["type"] == "multinomial":
                    prob = prob[c] ** float(X[idx])
                elif prob["type"] == "bernoulli":
                    prob = prob[c] if X[idx] else 1 - prob[c]
                elif prob["type"] == "categorical":
                    prob = prob[str(c)][X[idx]]
                else:
                    prob = (
                        1
                        / np.sqrt(2 * np.pi * prob[c]["sigma_sq"])
                        * np.exp(
                            -((float(X[idx]) - prob[c]["mu"]) ** 2)
                            / (2 * prob[c]["sigma_sq"])
                        )
                    )
                sub_result += [prob]
            res += [sub_result]
        res = np.array(res).prod(axis=1) * self.prior_
        if return_proba:
            return res / res.sum()
        else:
            res = np.argmax(res)
            res = self.classes_[res]
            return res

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts using the input matrix.

        Parameters
        ----------
        X: list / numpy.array
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Predicted values.
        """
        return np.apply_along_axis(self._predict_row, 1, X)

    def _predict_proba_row(self, X: ArrayLike) -> np.ndarray:
        """
        Predicts probablities for one row.
        """
        return self._predict_row(X, True)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Computes the model's probabilites using the input matrix.

        Parameters
        ----------
        X: list / numpy.array
            The data on which to make the prediction.

        Returns
        -------
        numpy.array
            Probabilities.
        """
        return np.apply_along_axis(self._predict_proba_row, 1, X)

    # Prediction / Transformation Methods - IN DATABASE.

    def _predict_score_sql(self, X: ArrayLike) -> list[str]:
        """
        Returns the final score for each class. The score
        divided to the total corresponds to the probability.
        """
        score = []
        for idx, c in enumerate(self.classes_):
            sub_result = []
            for idx2, x in enumerate(X):
                prob = self.attributes_[idx2]
                if prob["type"] == "multinomial":
                    prob = f"POWER({prob[c]}, {x})"
                elif prob["type"] == "bernoulli":
                    prob = f"(CASE WHEN {x} THEN {prob[c]} ELSE {1 - prob[c]} END)"
                elif prob["type"] == "categorical":
                    prob_res = f"DECODE({x}"
                    for cat in prob[str(c)]:
                        prob_res += f", '{cat}', {prob[str(c)][cat]}"
                    prob = prob_res + ")"
                else:
                    prob = f"""
                        {1 / np.sqrt(2 * np.pi * prob[c]['sigma_sq'])} 
                      * EXP(- POWER({x} - {prob[c]['mu']}, 2) 
                      / {2 * prob[c]['sigma_sq']})"""
                sub_result += [clean_query(prob)]
            score += [" * ".join(sub_result) + f" * {self.prior_[idx]}"]
        return score

    def predict_proba_sql(self, X: ArrayLike) -> list[str]:
        """
        Returns the SQL code needed to deploy the model probabilities
        using its attributes.

        Parameters
        ----------
        X: ArrayLike
            The names or values of the input predictors.

        Returns
        -------
        list
            SQL code.
        """
        score = self._predict_score_sql(X)
        score_sum = f"({' + '.join(score)})"
        return [f"({s}) / {score_sum}" for s in score]
