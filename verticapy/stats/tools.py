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
# Standard Python Modules
import math, decimal, datetime

# Other Python Modules
from scipy.stats import chi2, norm, f
import numpy as np

# VerticaPy Modules
import verticapy
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.learn.linear_model import LinearRegression
from verticapy import vDataFrame

# Statistical Tests & Tools
# ---#
def adfuller(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: list = [],
    p: int = 1,
    with_trend: bool = False,
    regresults: bool = False,
):
    """
---------------------------------------------------------------------------
Augmented Dickey Fuller test (Time Series stationarity).

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
column: str
    Input vcolumn to test.
ts: str
    vcolumn used as timeline. It will be to use to order the data. It can be
    a numerical or type date like (date, datetime, timestamp...) vcolumn.
by: list, optional
    vcolumns used in the partition.
p: int, optional
    Number of lags to consider in the test.
with_trend: bool, optional
    Adds a trend in the Regression.
regresults: bool, optional
    If True, the full regression results are returned.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """

    def critical_value(alpha, N, with_trend):
        if not (with_trend):
            if N <= 25:
                if alpha == 0.01:
                    return -3.75
                elif alpha == 0.10:
                    return -2.62
                elif alpha == 0.025:
                    return -3.33
                else:
                    return -3.00
            elif N <= 50:
                if alpha == 0.01:
                    return -3.58
                elif alpha == 0.10:
                    return -2.60
                elif alpha == 0.025:
                    return -3.22
                else:
                    return -2.93
            elif N <= 100:
                if alpha == 0.01:
                    return -3.51
                elif alpha == 0.10:
                    return -2.58
                elif alpha == 0.025:
                    return -3.17
                else:
                    return -2.89
            elif N <= 250:
                if alpha == 0.01:
                    return -3.46
                elif alpha == 0.10:
                    return -2.57
                elif alpha == 0.025:
                    return -3.14
                else:
                    return -2.88
            elif N <= 500:
                if alpha == 0.01:
                    return -3.44
                elif alpha == 0.10:
                    return -2.57
                elif alpha == 0.025:
                    return -3.13
                else:
                    return -2.87
            else:
                if alpha == 0.01:
                    return -3.43
                elif alpha == 0.10:
                    return -2.57
                elif alpha == 0.025:
                    return -3.12
                else:
                    return -2.86
        else:
            if N <= 25:
                if alpha == 0.01:
                    return -4.38
                elif alpha == 0.10:
                    return -3.24
                elif alpha == 0.025:
                    return -3.95
                else:
                    return -3.60
            elif N <= 50:
                if alpha == 0.01:
                    return -4.15
                elif alpha == 0.10:
                    return -3.18
                elif alpha == 0.025:
                    return -3.80
                else:
                    return -3.50
            elif N <= 100:
                if alpha == 0.01:
                    return -4.04
                elif alpha == 0.10:
                    return -3.15
                elif alpha == 0.025:
                    return -3.73
                else:
                    return -5.45
            elif N <= 250:
                if alpha == 0.01:
                    return -3.99
                elif alpha == 0.10:
                    return -3.13
                elif alpha == 0.025:
                    return -3.69
                else:
                    return -3.43
            elif N <= 500:
                if alpha == 0.01:
                    return 3.98
                elif alpha == 0.10:
                    return -3.13
                elif alpha == 0.025:
                    return -3.68
                else:
                    return -3.42
            else:
                if alpha == 0.01:
                    return -3.96
                elif alpha == 0.10:
                    return -3.12
                elif alpha == 0.025:
                    return -3.66
                else:
                    return -3.41

    check_types(
        [
            ("ts", ts, [str],),
            ("column", column, [str],),
            ("p", p, [int, float],),
            ("by", by, [list],),
            ("with_trend", with_trend, [bool],),
            ("regresults", regresults, [bool],),
            ("vdf", vdf, [vDataFrame,],),
        ],
    )
    columns_check([ts, column] + by, vdf)
    ts = vdf_columns_names([ts], vdf)[0]
    column = vdf_columns_names([column], vdf)[0]
    by = vdf_columns_names(by, vdf)
    schema = vdf._VERTICAPY_VARIABLES_["schema_writing"]
    if not (schema):
        schema = "public"
    name = "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(
        schema, gen_name([column]).upper()
    )
    relation_name = "{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_VIEW_{}".format(
        schema, gen_name([column]).upper()
    )
    try:
        vdf._VERTICAPY_VARIABLES_["cursor"].execute(
            "DROP MODEL IF EXISTS {}".format(name)
        )
        vdf._VERTICAPY_VARIABLES_["cursor"].execute(
            "DROP VIEW IF EXISTS {}".format(relation_name)
        )
    except:
        pass
    lag = [
        "LAG({}, 1) OVER ({}ORDER BY {}) AS lag1".format(
            column, "PARTITION BY {}".format(", ".join(by)) if (by) else "", ts
        )
    ]
    lag += [
        "LAG({}, {}) OVER ({}ORDER BY {}) - LAG({}, {}) OVER ({}ORDER BY {}) AS delta{}".format(
            column,
            i,
            "PARTITION BY {}".format(", ".join(by)) if (by) else "",
            ts,
            column,
            i + 1,
            "PARTITION BY {}".format(", ".join(by)) if (by) else "",
            ts,
            i,
        )
        for i in range(1, p + 1)
    ]
    lag += [
        "{} - LAG({}, 1) OVER ({}ORDER BY {}) AS delta".format(
            column, column, "PARTITION BY {}".format(", ".join(by)) if (by) else "", ts
        )
    ]
    query = "CREATE VIEW {} AS SELECT {}, {} AS ts FROM {}".format(
        relation_name,
        ", ".join(lag),
        "TIMESTAMPDIFF(SECOND, {}, MIN({}) OVER ())".format(ts, ts)
        if vdf[ts].isdate()
        else ts,
        vdf.__genSQL__(),
    )
    vdf._VERTICAPY_VARIABLES_["cursor"].execute(query)
    model = LinearRegression(
        name, vdf._VERTICAPY_VARIABLES_["cursor"], solver="Newton", max_iter=1000
    )
    predictors = ["lag1"] + ["delta{}".format(i) for i in range(1, p + 1)]
    if with_trend:
        predictors += ["ts"]
    model.fit(
        relation_name, predictors, "delta",
    )
    coef = model.coef_
    vdf._VERTICAPY_VARIABLES_["cursor"].execute("DROP MODEL IF EXISTS {}".format(name))
    vdf._VERTICAPY_VARIABLES_["cursor"].execute(
        "DROP VIEW IF EXISTS {}".format(relation_name)
    )
    if regresults:
        return coef
    coef = coef.transpose()
    DF = coef.values["lag1"][0] / (max(coef.values["lag1"][1], 1e-99))
    p_value = coef.values["lag1"][3]
    count = vdf.shape()[0]
    result = tablesample(
        {
            "index": [
                "ADF Test Statistic",
                "p_value",
                "# Lags used",
                "# Observations Used",
                "Critical Value (1%)",
                "Critical Value (2.5%)",
                "Critical Value (5%)",
                "Critical Value (10%)",
                "Stationarity (alpha = 1%)",
            ],
            "value": [
                DF,
                p_value,
                p,
                count,
                critical_value(0.01, count, with_trend),
                critical_value(0.025, count, with_trend),
                critical_value(0.05, count, with_trend),
                critical_value(0.10, count, with_trend),
                DF < critical_value(0.01, count, with_trend) and p_value < 0.01,
            ],
        }
    )
    return result


# ---#
def durbin_watson(
    vdf: vDataFrame, eps: str, ts: str, by: list = [],
):
    """
---------------------------------------------------------------------------
Durbin Watson test (residuals autocorrelation).

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
eps: str
    Input residual vcolumn.
ts: str
    vcolumn used as timeline. It will be to use to order the data. It can be
    a numerical or type date like (date, datetime, timestamp...) vcolumn.
by: list, optional
    vcolumns used in the partition.

Returns
-------
float
    Durbin Watson statistic
    """
    check_types(
        [
            ("ts", ts, [str],),
            ("eps", eps, [str],),
            ("by", by, [list],),
            ("vdf", vdf, [vDataFrame, str,],),
        ],
    )
    columns_check([eps] + [ts] + by, vdf)
    eps = vdf_columns_names([eps], vdf)[0]
    ts = vdf_columns_names([ts], vdf)[0]
    by = vdf_columns_names(by, vdf)
    query = "(SELECT et, LAG(et) OVER({}ORDER BY {}) AS lag_et FROM (SELECT {} AS et, {}{} FROM {}) VERTICAPY_SUBTABLE) VERTICAPY_SUBTABLE".format(
        "PARTITION BY {} ".format(", ".join(by)) if (by) else "",
        ts,
        eps,
        ts,
        (", " + ", ".join(by)) if (by) else "",
        vdf.__genSQL__(),
    )
    vdf.__executeSQL__(
        "SELECT SUM(POWER(et - lag_et, 2)) / SUM(POWER(et, 2)) FROM {}".format(query),
        title="Computes the Durbin Watson d.",
    )
    d = vdf._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
    return d


# ---#
def endogtest(
    vdf: vDataFrame, eps: str, X: list,
):
    """
---------------------------------------------------------------------------
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
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [("eps", eps, [str],), ("X", X, [list],), ("vdf", vdf, [vDataFrame, str,],),],
    )
    columns_check([eps] + X, vdf)
    eps = vdf_columns_names([eps], vdf)[0]
    X = vdf_columns_names(X, vdf)

    from verticapy.learn.linear_model import LinearRegression

    schema_writing = vdf._VERTICAPY_VARIABLES_["schema_writing"]
    if not (schema_writing):
        schema_writing = "public"
    name = schema_writing + ".VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(
        get_session(vdf._VERTICAPY_VARIABLES_["cursor"])
    )
    model = LinearRegression(name, cursor=vdf._VERTICAPY_VARIABLES_["cursor"])
    try:
        model.fit(vdf, X, eps)
        R2 = model.score("r2")
        model.drop()
    except:
        try:
            model.set_params({"solver": "bfgs"})
            model.fit(vdf, X, eps)
            R2 = model.score("r2")
            model.drop()
        except:
            model.drop()
            raise
    n = vdf.shape()[0]
    k = len(X)
    LM = n * R2
    lm_pvalue = chi2.sf(LM, k)
    F = (n - k - 1) * R2 / (1 - R2) / k
    f_pvalue = f.sf(F, k, n - k - 1)
    result = tablesample(
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


# ---#
def het_arch(
    vdf: vDataFrame, eps: str, ts: str, by: list = [], p: int = 1,
):
    """
---------------------------------------------------------------------------
Engle’s Test for Autoregressive Conditional Heteroscedasticity (ARCH).

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
eps: str
    Input residual vcolumn.
ts: str
    vcolumn used as timeline. It will be to use to order the data. It can be
    a numerical or type date like (date, datetime, timestamp...) vcolumn.
by: list, optional
    vcolumns used in the partition.
p: int, optional
    Number of lags to consider in the test.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("eps", eps, [str],),
            ("ts", ts, [str],),
            ("p", p, [int, float],),
            ("vdf", vdf, [vDataFrame, str,],),
        ],
    )
    columns_check([eps, ts] + by, vdf)
    eps = vdf_columns_names([eps], vdf)[0]
    ts = vdf_columns_names([ts], vdf)[0]
    by = vdf_columns_names(by, vdf)
    X = []
    X_names = []
    for i in range(0, p + 1):
        X += [
            "LAG(POWER({}, 2), {}) OVER({}ORDER BY {}) AS lag_{}".format(
                eps, i, ("PARTITION BY " + ", ".join(by)) if (by) else "", ts, i
            )
        ]
        X_names += ["lag_{}".format(i)]
    query = "(SELECT {} FROM {}) VERTICAPY_SUBTABLE".format(
        ", ".join(X), vdf.__genSQL__()
    )
    vdf_lags = vdf_from_relation(query, cursor=vdf._VERTICAPY_VARIABLES_["cursor"])
    from verticapy.learn.linear_model import LinearRegression

    schema_writing = vdf._VERTICAPY_VARIABLES_["schema_writing"]
    if not (schema_writing):
        schema_writing = "public"
    name = schema_writing + ".VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(
        get_session(vdf._VERTICAPY_VARIABLES_["cursor"])
    )
    model = LinearRegression(name, cursor=vdf._VERTICAPY_VARIABLES_["cursor"])
    try:
        model.fit(vdf_lags, X_names[1:], X_names[0])
        R2 = model.score("r2")
        model.drop()
    except:
        try:
            model.set_params({"solver": "bfgs"})
            model.fit(vdf_lags, X_names[1:], X_names[0])
            R2 = model.score("r2")
            model.drop()
        except:
            model.drop()
            raise
    n = vdf.shape()[0]
    k = len(X)
    LM = (n - p) * R2
    lm_pvalue = chi2.sf(LM, p)
    F = (n - 2 * p - 1) * R2 / (1 - R2) / p
    f_pvalue = f.sf(F, p, n - 2 * p - 1)
    result = tablesample(
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


# ---#
def het_breuschpagan(
    vdf: vDataFrame, eps: str, X: list,
):
    """
---------------------------------------------------------------------------
Breusch-Pagan test for heteroscedasticity.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
eps: str
    Input residual vcolumn.
X: list
    Exogenous Variables to test the heteroscedasticity on.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [("eps", eps, [str],), ("X", X, [list],), ("vdf", vdf, [vDataFrame, str,],),],
    )
    columns_check([eps] + X, vdf)
    eps = vdf_columns_names([eps], vdf)[0]
    X = vdf_columns_names(X, vdf)

    from verticapy.learn.linear_model import LinearRegression

    schema_writing = vdf._VERTICAPY_VARIABLES_["schema_writing"]
    if not (schema_writing):
        schema_writing = "public"
    name = schema_writing + ".VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(
        get_session(vdf._VERTICAPY_VARIABLES_["cursor"])
    )
    model = LinearRegression(name, cursor=vdf._VERTICAPY_VARIABLES_["cursor"])
    vdf_copy = vdf.copy()
    vdf_copy["VERTICAPY_TEMP_eps2"] = vdf_copy[eps] ** 2
    try:
        model.fit(vdf_copy, X, "VERTICAPY_TEMP_eps2")
        R2 = model.score("r2")
        model.drop()
    except:
        try:
            model.set_params({"solver": "bfgs"})
            model.fit(vdf_copy, X, "VERTICAPY_TEMP_eps2")
            R2 = model.score("r2")
            model.drop()
        except:
            model.drop()
            raise
    n = vdf.shape()[0]
    k = len(X)
    LM = n * R2
    lm_pvalue = chi2.sf(LM, k)
    F = (n - k - 1) * R2 / (1 - R2) / k
    f_pvalue = f.sf(F, k, n - k - 1)
    result = tablesample(
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


# ---#
def het_goldfeldquandt(
    vdf: vDataFrame, y: str, X: list, idx: int = 0, split: float = 0.5
):
    """
---------------------------------------------------------------------------
Goldfeld-Quandt homoscedasticity test.

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

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """

    def model_fit(input_relation, X, y, model):
        var = []
        for vdf_tmp in input_relation:
            model.drop()
            model.fit(vdf_tmp, X, y)
            model.predict(vdf_tmp, name="verticapy_prediction")
            vdf_tmp["residual_0"] = vdf_tmp[y] - vdf_tmp["verticapy_prediction"]
            var += [vdf_tmp["residual_0"].var()]
            model.drop()
        return var

    check_types(
        [
            ("y", y, [str],),
            ("X", X, [list],),
            ("idx", idx, [int, float],),
            ("split", split, [int, float],),
            ("vdf", vdf, [vDataFrame, str,],),
        ],
    )
    columns_check([y] + X, vdf)
    y = vdf_columns_names([y], vdf)[0]
    X = vdf_columns_names(X, vdf)
    split_value = vdf[X[idx]].quantile(split)
    vdf_0_half = vdf.search(vdf[X[idx]] < split_value)
    vdf_1_half = vdf.search(vdf[X[idx]] > split_value)
    from verticapy.learn.linear_model import LinearRegression

    schema_writing = vdf._VERTICAPY_VARIABLES_["schema_writing"]
    if not (schema_writing):
        schema_writing = "public"
    name = schema_writing + ".VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(
        get_session(vdf._VERTICAPY_VARIABLES_["cursor"])
    )
    model = LinearRegression(name, cursor=vdf._VERTICAPY_VARIABLES_["cursor"])
    try:
        var0, var1 = model_fit([vdf_0_half, vdf_1_half], X, y, model)
    except:
        try:
            model.set_params({"solver": "bfgs"})
            var0, var1 = model_fit([vdf_0_half, vdf_1_half], X, y, model)
        except:
            model.drop()
            raise
    n, m = vdf_0_half.shape()[0], vdf_1_half.shape()[0]
    F = var0 / var1
    f_pvalue = f.sf(F, n, m)
    result = tablesample({"index": ["F Value", "f_p_value",], "value": [F, f_pvalue],})
    return result


# ---#
def het_white(
    vdf: vDataFrame, eps: str, X: list,
):
    """
---------------------------------------------------------------------------
White’s Lagrange Multiplier Test for heteroscedasticity.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
eps: str
    Input residual vcolumn.
X: str
    Exogenous Variables to test the heteroscedasticity on.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [("eps", eps, [str],), ("X", X, [list],), ("vdf", vdf, [vDataFrame, str,],),],
    )
    columns_check([eps] + X, vdf)
    eps = vdf_columns_names([eps], vdf)[0]
    X = vdf_columns_names(X, vdf)
    X_0 = ["1"] + X
    variables = []
    variables_names = []
    for i in range(len(X_0)):
        for j in range(i, len(X_0)):
            if i != 0 or j != 0:
                variables += ["{} * {} AS var_{}_{}".format(X_0[i], X_0[j], i, j)]
                variables_names += ["var_{}_{}".format(i, j)]
    query = "(SELECT {}, POWER({}, 2) AS VERTICAPY_TEMP_eps2 FROM {}) VERTICAPY_SUBTABLE".format(
        ", ".join(variables), eps, vdf.__genSQL__()
    )
    vdf_white = vdf_from_relation(query, cursor=vdf._VERTICAPY_VARIABLES_["cursor"])

    from verticapy.learn.linear_model import LinearRegression

    schema_writing = vdf._VERTICAPY_VARIABLES_["schema_writing"]
    if not (schema_writing):
        schema_writing = "public"
    name = schema_writing + ".VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(
        get_session(vdf._VERTICAPY_VARIABLES_["cursor"])
    )
    model = LinearRegression(name, cursor=vdf._VERTICAPY_VARIABLES_["cursor"])
    try:
        model.fit(vdf_white, variables_names, "VERTICAPY_TEMP_eps2")
        R2 = model.score("r2")
        model.drop()
    except:
        try:
            model.set_params({"solver": "bfgs"})
            model.fit(vdf_white, variables_names, "VERTICAPY_TEMP_eps2")
            R2 = model.score("r2")
            model.drop()
        except:
            model.drop()
            raise
    n = vdf.shape()[0]
    if len(X) > 1:
        k = 2 * len(X) + math.factorial(len(X)) / 2 / (math.factorial(len(X) - 2))
    else:
        k = 1
    LM = n * R2
    lm_pvalue = chi2.sf(LM, k)
    F = (n - k - 1) * R2 / (1 - R2) / k
    f_pvalue = f.sf(F, k, n - k - 1)
    result = tablesample(
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


# ---#
def jarque_bera(vdf: vDataFrame, column: str, alpha: float = 0.05):
    """
---------------------------------------------------------------------------
Jarque-Bera test (Distribution Normality).

Parameters
----------
vdf: vDataFrame
    input vDataFrame.
column: str
    Input vcolumn to test.
alpha: float, optional
    Significance Level. Probability to accept H0.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("column", column, [str],),
            ("alpha", alpha, [int, float],),
            ("vdf", vdf, [vDataFrame,],),
        ],
    )
    columns_check([column], vdf)
    column = vdf_columns_names([column], vdf)[0]
    jb, kurtosis, skewness, n = (
        vdf[column].agg(["jb", "kurtosis", "skewness", "count"]).values[column]
    )
    pvalue = chi2.sf(jb, 2)
    result = False if pvalue < alpha else True
    result = tablesample(
        {
            "index": [
                "Jarque Bera Test Statistic",
                "p_value",
                "# Observations Used",
                "Kurtosis - 3",
                "Skewness",
                "Distribution Normality",
            ],
            "value": [jb, pvalue, n, kurtosis, skewness, result],
        }
    )
    return result


# ---#
def kurtosistest(vdf: vDataFrame, column: str):
    """
---------------------------------------------------------------------------
Test whether the kurtosis is different from the normal distribution.

Parameters
----------
vdf: vDataFrame
    input vDataFrame.
column: str
    Input vcolumn to test.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types([("column", column, [str],), ("vdf", vdf, [vDataFrame,],),],)
    columns_check([column], vdf)
    column = vdf_columns_names([column], vdf)[0]
    g2, n = vdf[column].agg(["kurtosis", "count"]).values[column]
    mu1 = -6 / (n + 1)
    mu2 = 24 * n * (n - 2) * (n - 3) / (((n + 1) ** 2) * (n + 3) * (n + 5))
    gamma1 = (
        6
        * (n ** 2 - 5 * n + 2)
        / ((n + 7) * (n + 9))
        * math.sqrt(6 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3)))
    )
    A = 6 + 8 / gamma1 * (2 / gamma1 + math.sqrt(1 + 4 / (gamma1 ** 2)))
    B = (1 - 2 / A) / (1 + (g2 - mu1) / math.sqrt(mu2) * math.sqrt(2 / (A - 4)))
    B = B ** (1 / 3) if B > 0 else (-B) ** (1 / 3)
    Z2 = math.sqrt(9 * A / 2) * (1 - 2 / (9 * A) - B)
    pvalue = 2 * norm.sf(abs(Z2))
    result = tablesample({"index": ["Statistic", "p_value",], "value": [Z2, pvalue],})
    return result


# ---#
def ljungbox(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: list = [],
    p: int = 1,
    alpha: float = 0.05,
    box_pierce: bool = False,
):
    """
---------------------------------------------------------------------------
Ljung–Box test (whether any of a group of autocorrelations of a time series 
are different from zero).

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
column: str
    Input vcolumn to test.
ts: str
    vcolumn used as timeline. It will be to use to order the data. It can be
    a numerical or type date like (date, datetime, timestamp...) vcolumn.
by: list, optional
    vcolumns used in the partition.
p: int, optional
    Number of lags to consider in the test.
alpha: float, optional
    Significance Level. Probability to accept H0.
box_pierce: bool
    If set to True, the Box-Pierce statistic will be used.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("ts", ts, [str],),
            ("column", column, [str],),
            ("by", by, [list],),
            ("p", p, [int, float],),
            ("alpha", alpha, [int, float],),
            ("box_pierce", box_pierce, [bool],),
            ("vdf", vdf, [vDataFrame,],),
        ],
    )
    columns_check([column] + [ts] + by, vdf)
    column = vdf_columns_names([column], vdf)[0]
    ts = vdf_columns_names([ts], vdf)[0]
    by = vdf_columns_names(by, vdf)
    acf = vdf.acf(column=column, ts=ts, by=by, p=p, show=False)
    if p >= 2:
        acf = acf.values["value"]
    else:
        acf = [acf]
    n = vdf[column].count()
    name = (
        "Ljung–Box Test Statistic" if not (box_pierce) else "Box-Pierce Test Statistic"
    )
    result = tablesample(
        {"index": [], name: [], "p_value": [], "Serial Correlation": []}
    )
    Q = 0
    for k in range(p):
        div = n - k - 1 if not (box_pierce) else 1
        mult = n * (n + 2) if not (box_pierce) else n
        Q += mult * acf[k] ** 2 / div
        pvalue = chi2.sf(Q, k + 1)
        result.values["index"] += [k + 1]
        result.values[name] += [Q]
        result.values["p_value"] += [pvalue]
        result.values["Serial Correlation"] += [True if pvalue < alpha else False]
    return result


# ---#
def mkt(vdf: vDataFrame, column: str, ts: str, alpha: float = 0.05):
    """
---------------------------------------------------------------------------
Mann Kendall test (Time Series trend).

\u26A0 Warning : This Test is computationally expensive. It is using a CROSS 
                 JOIN during the computation. The complexity is O(n * k), n 
                 being the total count of the vDataFrame and k the number
                 of rows to use to do the test.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
column: str
    Input vcolumn to test.
ts: str
    vcolumn used as timeline. It will be to use to order the data. It can be
    a numerical or type date like (date, datetime, timestamp...) vcolumn.
alpha: float, optional
    Significance Level. Probability to accept H0.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("ts", ts, [str],),
            ("column", column, [str],),
            ("alpha", alpha, [int, float],),
            ("vdf", vdf, [vDataFrame,],),
        ],
    )
    columns_check([column, ts], vdf)
    column = vdf_columns_names([column], vdf)[0]
    ts = vdf_columns_names([ts], vdf)[0]
    table = "(SELECT {}, {} FROM {})".format(column, ts, vdf.__genSQL__())
    query = "SELECT SUM(SIGN(y.{} - x.{})) FROM {} x CROSS JOIN {} y WHERE y.{} > x.{}".format(
        column, column, table, table, ts, ts
    )
    vdf.__executeSQL__(query, title="Computes the Mann Kendall S.")
    S = vdf._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
    try:
        S = float(S)
    except:
        S = None
    n = vdf[column].count()
    query = "SELECT SQRT(({} * ({} - 1) * (2 * {} + 5) - SUM(row * (row - 1) * (2 * row + 5))) / 18) FROM (SELECT row FROM (SELECT ROW_NUMBER() OVER (PARTITION BY {}) AS row FROM {}) VERTICAPY_SUBTABLE GROUP BY row) VERTICAPY_SUBTABLE".format(
        n, n, n, column, vdf.__genSQL__()
    )
    vdf.__executeSQL__(query, title="Computes the Mann Kendall S standard deviation.")
    STDS = vdf._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
    try:
        STDS = float(STDS)
    except:
        STDS = None
    if STDS in (None, 0) or S == None:
        return None
    if S > 0:
        ZMK = (S - 1) / STDS
        trend = "increasing"
    elif S < 0:
        ZMK = (S + 1) / STDS
        trend = "decreasing"
    else:
        ZMK = 0
        trend = "no trend"
    pvalue = 2 * norm.sf(abs(ZMK))
    result = (
        True
        if (ZMK <= 0 and pvalue < alpha) or (ZMK >= 0 and pvalue < alpha)
        else False
    )
    if not (result):
        trend = "no trend"
    result = tablesample(
        {
            "index": [
                "Mann Kendall Test Statistic",
                "S",
                "STDS",
                "p_value",
                "Monotonic Trend",
                "Trend",
            ],
            "value": [ZMK, S, STDS, pvalue, result, trend],
        }
    )
    return result


# ---#
def normaltest(vdf: vDataFrame, column: str):
    """
---------------------------------------------------------------------------
Test whether a sample differs from a normal distribution.

Parameters
----------
vdf: vDataFrame
    input vDataFrame.
column: str
    Input vcolumn to test.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    Z1, Z2 = skewtest(vdf, column)["value"][0], kurtosistest(vdf, column)["value"][0]
    Z = Z1 ** 2 + Z2 ** 2
    pvalue = chi2.sf(Z, 2)
    result = tablesample({"index": ["Statistic", "p_value",], "value": [Z, pvalue],})
    return result


# ---#
def seasonal_decompose(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: list = [],
    period: int = -1,
    polynomial_order: int = 1,
    estimate_seasonality: bool = True,
    rule: (str, datetime.timedelta) = None,
    mult: bool = False,
    two_sided: bool = False,
):
    """
---------------------------------------------------------------------------
Performs a seasonal time series decomposition.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
column: str
    Input vcolumn to decompose.
ts: str
    TS (Time Series) vcolumn to use to order the data. It can be of type date
    or a numerical vcolumn.
by: list, optional
    vcolumns used in the partition.
period: int, optional
	Time Series period. It is used to retrieve the seasonality component.
    if period <= 0, the seasonal component will be estimated using ACF. In 
    this case, polynomial_order must be greater than 0.
polynomial_order: int, optional
    If greater than 0, the trend will be estimated using a polynomial of degree
    'polynomial_order'. The parameter 'two_sided' will be ignored.
    If equal to 0, the trend will be estimated using Moving Averages.
estimate_seasonality: bool, optional
    If set to True, the seasonality will be estimated using cosine and sine
    functions.
rule: str / time, optional
    Interval to use to slice the time. For example, '5 minutes' will create records
    separated by '5 minutes' time interval.
mult: bool, optional
	If set to True, the decomposition type will be 'multiplicative'. Otherwise,
	it is 'additive'.
two_sided: bool, optional
    If set to True, a centered moving average is used for the trend isolation.
    Otherwise only past values are used.

Returns
-------
vDataFrame
    object containing (ts, column, TS seasonal part, TS trend, TS noise).
    """
    if isinstance(by, str):
        by = [by]
    check_types(
        [
            ("ts", ts, [str],),
            ("column", column, [str],),
            ("by", by, [list],),
            ("rule", rule, [str, datetime.timedelta,],),
            ("vdf", vdf, [vDataFrame,],),
            ("period", period, [int,],),
            ("mult", mult, [bool,],),
            ("two_sided", two_sided, [bool,],),
            ("polynomial_order", polynomial_order, [int,],),
            ("estimate_seasonality", estimate_seasonality, [bool,],),
        ],
    )
    assert period > 0 or polynomial_order > 0, ParameterError("Parameters 'polynomial_order' and 'period' can not be both null.")
    columns_check([column, ts] + by, vdf)
    ts, column, by = (
        vdf_columns_names([ts], vdf)[0],
        vdf_columns_names([column], vdf)[0],
        vdf_columns_names(by, vdf),
    )
    if rule:
        vdf_tmp = vdf.asfreq(ts=ts, rule=period, method={column: "linear"}, by=by)
    else:
        vdf_tmp = vdf[[ts, column]]
    trend_name, seasonal_name, epsilon_name = (
        "{}_trend".format(column[1:-1]),
        "{}_seasonal".format(column[1:-1]),
        "{}_epsilon".format(column[1:-1]),
    )
    by, by_tmp = "" if not (by) else "PARTITION BY " + ", ".join(vdf_columns_names(by, self)) + " ", by
    if polynomial_order <= 0:
        if two_sided:
            if period == 1:
                window = (-1, 1)
            else:
                if period % 2 == 0:
                    window = (-period / 2 + 1, period / 2)
                else:
                    window = (int(-period / 2), int(period / 2))
        else:
            if period == 1:
                window = (-2, 0)
            else:
                window = (-period + 1, 0)
        vdf_tmp.rolling("avg", window, column, by_tmp, ts, trend_name)
    else:
        vdf_poly = vdf_tmp.copy()
        X = []
        for i in range(1, polynomial_order + 1):
            vdf_poly[f"t_{i}"] = f"POWER(ROW_NUMBER() OVER ({by}ORDER BY {ts}), {i})"
            X += [f"t_{i}"]
        schema = vdf_poly._VERTICAPY_VARIABLES_["schema_writing"]
        if not (schema):
            schema = vdf_poly._VERTICAPY_VARIABLES_["schema"]
        if not (schema):
            schema = "public"

        from verticapy.learn.linear_model import LinearRegression
        model = LinearRegression(name="{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(schema, get_session(vdf_poly._VERTICAPY_VARIABLES_["cursor"])),
                                 cursor=vdf_poly._VERTICAPY_VARIABLES_["cursor"],
                                 solver="bfgs",
                                 max_iter=100,
                                 tol=1e-6,)
        model.drop()
        model.fit(vdf_poly, X, column)
        coefficients = model.coef_["coefficient"]
        coefficients = [str(coefficients[0])] + [f"{coefficients[i]} * POWER(ROW_NUMBER() OVER({by}ORDER BY {ts}), {i})" if i != 1 else f"{coefficients[1]} * ROW_NUMBER() OVER({by}ORDER BY {ts})" for i in range(1, polynomial_order + 1)]
        vdf_tmp[trend_name] = " + ".join(coefficients)
        model.drop()
    if mult:
        vdf_tmp[seasonal_name] = f'{column} / NULLIFZERO("{trend_name}")'
    else:
        vdf_tmp[seasonal_name] = vdf_tmp[column] - vdf_tmp[trend_name]
    if period <= 0:
        acf = vdf_tmp.acf(column=seasonal_name, ts=ts, p=23, acf_type="heatmap", show=False)
        period = int(acf["index"][1].split("_")[1])
        if period == 1:
            period = int(acf["index"][2].split("_")[1])
    vdf_tmp["row_number_id"] = f"MOD(ROW_NUMBER() OVER ({by} ORDER BY {ts}), {period})"
    if mult:
        vdf_tmp[
            seasonal_name
        ] = f"AVG({seasonal_name}) OVER (PARTITION BY row_number_id) / NULLIFZERO(AVG({seasonal_name}) OVER ())"
    else:
        vdf_tmp[
            seasonal_name
        ] = f"AVG({seasonal_name}) OVER (PARTITION BY row_number_id) - AVG({seasonal_name}) OVER ()"
    if estimate_seasonality:
        vdf_seasonality = vdf_tmp.copy()
        vdf_seasonality["t_cos"] = f"COS(2 * PI() * ROW_NUMBER() OVER ({by}ORDER BY {ts}) / {period})"
        vdf_seasonality["t_sin"] = f"SIN(2 * PI() * ROW_NUMBER() OVER ({by}ORDER BY {ts}) / {period})"
        X = ["t_cos", "t_sin",]
        schema = vdf_seasonality._VERTICAPY_VARIABLES_["schema_writing"]
        if not (schema):
            schema = vdf_seasonality._VERTICAPY_VARIABLES_["schema"]
        if not (schema):
            schema = "public"

        from verticapy.learn.linear_model import LinearRegression
        model = LinearRegression(name="{}.VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(schema, get_session(vdf_seasonality._VERTICAPY_VARIABLES_["cursor"])),
                                 cursor=vdf_seasonality._VERTICAPY_VARIABLES_["cursor"],
                                 solver="bfgs",
                                 max_iter=100,
                                 tol=1e-6,)
        model.drop()
        model.fit(vdf_seasonality, X, seasonal_name)
        coefficients = model.coef_["coefficient"]
        vdf_tmp[seasonal_name] = f"{coefficients[0]} + {coefficients[1]} * COS(2 * PI() * ROW_NUMBER() OVER ({by}ORDER BY {ts}) / {period}) + {coefficients[2]} * SIN(2 * PI() * ROW_NUMBER() OVER ({by}ORDER BY {ts}) / {period})"
        model.drop()
    if mult:
        vdf_tmp[
            epsilon_name
        ] = f'{column} / NULLIFZERO("{trend_name}") / NULLIFZERO("{seasonal_name}")'
    else:
        vdf_tmp[epsilon_name] = (
            vdf_tmp[column] - vdf_tmp[trend_name] - vdf_tmp[seasonal_name]
        )
    vdf_tmp["row_number_id"].drop()
    return vdf_tmp


# ---#
def skewtest(vdf: vDataFrame, column: str):
    """
---------------------------------------------------------------------------
Test whether the skewness is different from the normal distribution.

Parameters
----------
vdf: vDataFrame
    input vDataFrame.
column: str
    Input vcolumn to test.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types([("column", column, [str],), ("vdf", vdf, [vDataFrame,],),],)
    columns_check([column], vdf)
    column = vdf_columns_names([column], vdf)[0]
    g1, n = vdf[column].agg(["skewness", "count"]).values[column]
    mu1 = 0
    mu2 = 6 * (n - 2) / ((n + 1) * (n + 3))
    gamma1 = 0
    gamma2 = (
        36 * (n - 7) * (n ** 2 + 2 * n - 5) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    )
    W2 = math.sqrt(2 * gamma2 + 4) - 1
    delta = 1 / math.sqrt(math.log(math.sqrt(W2)))
    alpha2 = 2 / (W2 - 1)
    Z1 = delta * math.asinh(g1 / math.sqrt(alpha2 * mu2))
    pvalue = 2 * norm.sf(abs(Z1))
    result = tablesample({"index": ["Statistic", "p_value",], "value": [Z1, pvalue],})
    return result


# ---#
def variance_inflation_factor(
    vdf: vDataFrame, X: list, X_idx: int = None,
):
    """
---------------------------------------------------------------------------
Computes the variance inflation factor (VIF). It can be used to detect
multicollinearity in an OLS Regression Analysis.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
X: list
    Input Variables.
X_idx: int
    Index of the exogenous variable in X. If left to None, a tablesample will
    be returned with all the variables VIF.

Returns
-------
float
    VIF.
    """
    check_types(
        [
            ("X_idx", X_idx, [int],),
            ("X", X, [list],),
            ("vdf", vdf, [vDataFrame, str,],),
        ],
    )
    columns_check(X, vdf)
    X = vdf_columns_names(X, vdf)

    if isinstance(X_idx, str):
        columns_check([X_idx], vdf)
        for i in range(len(X)):
            if str_column(X[i]) == str_column(X_idx):
                X_idx = i
                break
    if isinstance(X_idx, (int, float)):
        X_r = []
        for i in range(len(X)):
            if i != X_idx:
                X_r += [X[i]]
        y_r = X[X_idx]

        from verticapy.learn.linear_model import LinearRegression

        schema_writing = vdf._VERTICAPY_VARIABLES_["schema_writing"]
        if not (schema_writing):
            schema_writing = "public"
        name = schema_writing + ".VERTICAPY_TEMP_MODEL_LINEAR_REGRESSION_{}".format(
            get_session(vdf._VERTICAPY_VARIABLES_["cursor"])
        )
        model = LinearRegression(name, cursor=vdf._VERTICAPY_VARIABLES_["cursor"])
        try:
            model.fit(vdf, X_r, y_r)
            R2 = model.score("r2")
            model.drop()
        except:
            try:
                model.set_params({"solver": "bfgs"})
                model.fit(vdf, X_r, y_r)
                R2 = model.score("r2")
                model.drop()
            except:
                model.drop()
                raise
        if 1 - R2 != 0:
            return 1 / (1 - R2)
        else:
            return np.inf
    elif X_idx == None:
        VIF = []
        for i in range(len(X)):
            VIF += [variance_inflation_factor(vdf, X, i)]
        return tablesample({"X_idx": X, "VIF": VIF})
    else:
        raise ParameterError(
            f"Wrong type for Parameter X_idx.\nExpected integer, found {type(X_idx)}."
        )
