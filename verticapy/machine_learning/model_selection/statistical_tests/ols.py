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
import math
from typing import Literal
import numpy as np
from scipy.stats import chi2, f

from verticapy._utils._sql._collect import save_verticapy_logs
import verticapy._config.config as conf
from verticapy._utils._gen import gen_tmp_name

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.linear_model import LinearRegression


@save_verticapy_logs
def endogtest(vdf: vDataFrame, eps: str, X: list):
    """
Endogeneity test.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
eps: str
    Input residual vcolumn.
X: list
    Input Variables to test the endogeneity on.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    eps, X = vdf._format_colnames(eps, X)
    name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="linear_reg")
    model = LinearRegression(name)
    try:
        model.fit(vdf, X, eps)
        R2 = model.score("r2")
    except:
        model.set_params({"solver": "bfgs"})
        model.fit(vdf, X, eps)
        R2 = model.score("r2")
    finally:
        model.drop()
    n = vdf.shape()[0]
    k = len(X)
    LM = n * R2
    lm_pvalue = chi2.sf(LM, k)
    F = (n - k - 1) * R2 / (1 - R2) / k
    f_pvalue = f.sf(F, k, n - k - 1)
    result = TableSample(
        {
            "index": [
                "Lagrange Multiplier Statistic",
                "lm_p_value",
                "F Value",
                "f_p_value",
            ],
            "value": [LM, lm_pvalue, F, f_pvalue],
        }
    )
    return result


@save_verticapy_logs
def het_breuschpagan(vdf: vDataFrame, eps: str, X: list):
    """
Uses the Breusch-Pagan to test a model for Heteroscedasticity.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
eps: str
    Input residual vDataColumn.
X: list
    The exogenous variables to test.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    eps, X = vdf._format_colnames(eps, X)
    name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="linear_reg")
    model = LinearRegression(name)
    vdf_copy = vdf.copy()
    vdf_copy["v_eps2"] = vdf_copy[eps] ** 2
    try:
        model.fit(vdf_copy, X, "v_eps2")
        R2 = model.score("r2")
    except:
        model.set_params({"solver": "bfgs"})
        model.fit(vdf_copy, X, "v_eps2")
        R2 = model.score("r2")
    finally:
        model.drop()
    n = vdf.shape()[0]
    k = len(X)
    LM = n * R2
    lm_pvalue = chi2.sf(LM, k)
    F = (n - k - 1) * R2 / (1 - R2) / k
    f_pvalue = f.sf(F, k, n - k - 1)
    result = TableSample(
        {
            "index": [
                "Lagrange Multiplier Statistic",
                "lm_p_value",
                "F Value",
                "f_p_value",
            ],
            "value": [LM, lm_pvalue, F, f_pvalue],
        }
    )
    return result


@save_verticapy_logs
def het_goldfeldquandt(
    vdf: vDataFrame,
    y: str,
    X: list,
    idx: int = 0,
    split: float = 0.5,
    alternative: Literal["increasing", "decreasing", "two-sided"] = "increasing",
):
    """
Goldfeld-Quandt Homoscedasticity test.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
y: str
    Response Column.
X: list
    Exogenous Variables.
idx: int, optional
    Column index of variable according to which observations are sorted 
    for the split.
split: float, optional
    Float to indicate where to split (Example: 0.5 to split on the median).
alternative: str, optional
    Specifies the alternative hypothesis for the p-value calculation,
    one of the following variances: "increasing", "decreasing", "two-sided".

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """

    def model_fit(input_relation, X, y, model):
        mse = []
        for vdf_tmp in input_relation:
            model.drop()
            model.fit(vdf_tmp, X, y)
            mse += [model.score(method="mse")]
            model.drop()
        return mse

    y, X = vdf._format_colnames(y, X)
    split_value = vdf[X[idx]].quantile(split)
    vdf_0_half = vdf.search(vdf[X[idx]] < split_value)
    vdf_1_half = vdf.search(vdf[X[idx]] > split_value)
    name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="linear_reg")
    model = LinearRegression(name)
    try:
        mse0, mse1 = model_fit([vdf_0_half, vdf_1_half], X, y, model)
    except:
        model.set_params({"solver": "bfgs"})
        mse0, mse1 = model_fit([vdf_0_half, vdf_1_half], X, y, model)
    finally:
        model.drop()
    n, m, k = vdf_0_half.shape()[0], vdf_1_half.shape()[0], len(X)
    F = mse1 / mse0
    if alternative.lower() in ["increasing"]:
        f_pvalue = f.sf(F, n - k, m - k)
    elif alternative.lower() in ["decreasing"]:
        f_pvalue = f.sf(1.0 / F, m - k, n - k)
    elif alternative.lower() in ["two-sided"]:
        fpval_sm = f.cdf(F, m - k, n - k)
        fpval_la = f.sf(F, m - k, n - k)
        f_pvalue = 2 * min(fpval_sm, fpval_la)
    result = TableSample({"index": ["F Value", "f_p_value"], "value": [F, f_pvalue]})
    return result


@save_verticapy_logs
def het_white(vdf: vDataFrame, eps: str, X: list):
    """
White’s Lagrange Multiplier Test for Heteroscedasticity.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
eps: str
    Input residual vcolumn.
X: str
    Exogenous Variables to test the Heteroscedasticity on.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    eps, X = vdf._format_colnames(eps, X)
    X_0 = ["1"] + X
    variables = []
    variables_names = []
    for i in range(len(X_0)):
        for j in range(i, len(X_0)):
            if i != 0 or j != 0:
                variables += ["{} * {} AS var_{}_{}".format(X_0[i], X_0[j], i, j)]
                variables_names += ["var_{}_{}".format(i, j)]
    query = "SELECT {}, POWER({}, 2) AS v_eps2 FROM {}".format(
        ", ".join(variables), eps, vdf._genSQL()
    )
    vdf_white = vDataFrame(query)
    name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="linear_reg")
    model = LinearRegression(name)
    try:
        model.fit(vdf_white, variables_names, "v_eps2")
        R2 = model.score("r2")
    except:
        model.set_params({"solver": "bfgs"})
        model.fit(vdf_white, variables_names, "v_eps2")
        R2 = model.score("r2")
    finally:
        model.drop()
    n = vdf.shape()[0]
    if len(X) > 1:
        k = 2 * len(X) + math.factorial(len(X)) / 2 / (math.factorial(len(X) - 2))
    else:
        k = 1
    LM = n * R2
    lm_pvalue = chi2.sf(LM, k)
    F = (n - k - 1) * R2 / (1 - R2) / k
    f_pvalue = f.sf(F, k, n - k - 1)
    result = TableSample(
        {
            "index": [
                "Lagrange Multiplier Statistic",
                "lm_p_value",
                "F Value",
                "f_p_value",
            ],
            "value": [LM, lm_pvalue, F, f_pvalue],
        }
    )
    return result


@save_verticapy_logs
def variance_inflation_factor(vdf: vDataFrame, X: list, X_idx: int = None):
    """
Computes the variance inflation factor (VIF). It can be used to detect
multicollinearity in an OLS Regression Analysis.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
X: list
    Input Variables.
X_idx: int
    Index of the exogenous variable in X. If left to None, a TableSample will
    be returned with all the variables VIF.

Returns
-------
float
    VIF.
    """
    X, X_idx = vdf._format_colnames(X, X_idx)

    if isinstance(X_idx, str):
        for i in range(len(X)):
            if X[i] == X_idx:
                X_idx = i
                break
    if isinstance(X_idx, (int, float)):
        X_r = []
        for i in range(len(X)):
            if i != X_idx:
                X_r += [X[i]]
        y_r = X[X_idx]
        name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="linear_reg")
        model = LinearRegression(name)
        try:
            model.fit(vdf, X_r, y_r)
            R2 = model.score("r2")
        except:
            model.set_params({"solver": "bfgs"})
            model.fit(vdf, X_r, y_r)
            R2 = model.score("r2")
        finally:
            model.drop()
        if 1 - R2 != 0:
            return 1 / (1 - R2)
        else:
            return np.inf
    elif X_idx == None:
        VIF = []
        for i in range(len(X)):
            VIF += [variance_inflation_factor(vdf, X, i)]
        return TableSample({"X_idx": X, "VIF": VIF})
    else:
        raise ParameterError(
            f"Wrong type for Parameter X_idx.\nExpected integer, found {type(X_idx)}."
        )
