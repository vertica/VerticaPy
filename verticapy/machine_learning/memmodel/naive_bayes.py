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
import numpy as np
from typing import Union

# VerticaPy Modules
from verticapy.sql._utils import clean_query


def predict_from_nb(
    X: Union[list, np.ndarray],
    attributes: list,
    classes: Union[list, np.ndarray],
    prior: Union[list, np.ndarray],
    return_proba: bool = False,
) -> np.ndarray:
    """
    Predicts using a naive Bayes model and the input attributes.

    Parameters
    ----------
    X: list / numpy.array
        Data on which to make the prediction.
    attributes: list
        List of the model's attributes. Each feature must be represented
        by a dictionary, which differs based on the distribution.
          For 'gaussian':
            Key 'type' must have as value 'gaussian'.
            Each of the model's classes must include a dictionary with two keys:
              sigma_sq: Square root of the standard deviation.
              mu: Average.
            Example: {'type': 'gaussian', 
                      'C': {'mu': 63.9878308300395, 'sigma_sq': 7281.87598377196}, 
                      'Q': {'mu': 13.0217386792453, 'sigma_sq': 211.626862330204}, 
                      'S': {'mu': 27.6928120412844, 'sigma_sq': 1428.57067393938}}
          For 'multinomial':
            Key 'type' must have as value 'multinomial'.
            Each of the model's classes must be represented by a key with its
            probability as the value.
            Example: {'type': 'multinomial', 
                      'C': 0.771666666666667, 
                      'Q': 0.910714285714286, 
                      'S': 0.878216123499142}
          For 'bernoulli':
            Key 'type' must have as value 'bernoulli'.
            Each of the model's classes must be represented by a key with its
            probability as the value.
            Example: {'type': 'bernoulli', 
                      'C': 0.537254901960784, 
                      'Q': 0.277777777777778, 
                      'S': 0.324942791762014}
          For 'categorical':
            Key 'type' must have as value 'categorical'.
            Each of the model's classes must include a dictionary with all the feature
            categories.
            Example: {'type': 'categorical', 
                      'C': {'female': 0.407843137254902, 'male': 0.592156862745098}, 
                      'Q': {'female': 0.416666666666667, 'male': 0.583333333333333}, 
                      'S': {'female': 0.311212814645309, 'male': 0.688787185354691}}
    classes: list / numpy.array
        The classes for the naive Bayes model.
    prior: list / numpy.array
        The model's classes probabilities.
    return_proba: bool, optional
        If set to True and the method is set to 'LogisticRegression' or 'LinearSVC', 
        the probability is returned.

    Returns
    -------
    numpy.array
        Predicted values
    """

    def naive_bayes_score_row(X):
        result = []
        for c in classes:
            sub_result = []
            for idx, elem in enumerate(X):
                prob = attributes[idx]
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
            result += [sub_result]
        result = np.array(result).prod(axis=1) * prior
        if return_proba:
            return result / result.sum()
        else:
            return classes[np.argmax(result)]

    return np.apply_along_axis(naive_bayes_score_row, 1, X)


def sql_from_nb(
    X: Union[list, np.ndarray],
    attributes: list,
    classes: Union[list, np.ndarray],
    prior: Union[list, np.ndarray],
) -> list:
    """
    Predicts using a naive Bayes model and the input attributes. This function
    returns the unnormalized probabilities of each class as raw SQL code to 
    deploy the model.

    Parameters
    ----------
    X: list / numpy.array
        Data on which to make the prediction.
    attributes: list
        List of the model's attributes. Each feature is respresented a dictionary,
        the contents of which differs for each distribution type.
          For 'gaussian':
            Key 'type' must have the value 'gaussian'.
            Each of the model's classes must include a dictionary with two keys:
              sigma_sq: Square root of the standard deviation.
              mu: Average.
            Example: {'type': 'gaussian', 
                      'C': {'mu': 63.9878308300395, 'sigma_sq': 7281.87598377196}, 
                      'Q': {'mu': 13.0217386792453, 'sigma_sq': 211.626862330204}, 
                      'S': {'mu': 27.6928120412844, 'sigma_sq': 1428.57067393938}}
          For 'multinomial':
            Key 'type' must have the value 'multinomial'.
            Each of the model's classes must be represented by a key with its 
            probability as the value.
            Example: {'type': 'multinomial', 
                      'C': 0.771666666666667, 
                      'Q': 0.910714285714286, 
                      'S': 0.878216123499142}
          For 'bernoulli':
            Key 'type' must have the value 'bernoulli'.
            Each of the model's classes must be represented by a key with its 
            probability as the value.
            Example: {'type': 'bernoulli', 
                      'C': 0.537254901960784, 
                      'Q': 0.277777777777778, 
                      'S': 0.324942791762014}
          For 'categorical':
            Key 'type' must have the value 'categorical'.
            Each of the model's classes must include a dictionary with all the 
            feature categories.
            Example: {'type': 'categorical', 
                      'C': {'female': 0.407843137254902, 'male': 0.592156862745098}, 
                      'Q': {'female': 0.416666666666667, 'male': 0.583333333333333}, 
                      'S': {'female': 0.311212814645309, 'male': 0.688787185354691}}
    classes: list / numpy.array
        The classes for the naive bayes model.
    prior: list / numpy.array
        The model's classes probabilities.

    Returns
    -------
    numpy.array
        Predicted values
    """
    result = []
    for idx, c in enumerate(classes):
        sub_result = []
        for idx2, x in enumerate(X):
            prob = attributes[idx2]
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
        result += [" * ".join(sub_result) + f" * {prior[idx]}"]
    return result
