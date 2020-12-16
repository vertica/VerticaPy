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
from verticapy.utilities import *
from verticapy.toolbox import *
from verticapy.learn.linear_model import LinearRegression
from verticapy import vDataFrame

# Other modules
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

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
    vdf: vDataFrame, column: str, ts: str, X: list, by: list = [],
):
    """
---------------------------------------------------------------------------
Durbin Watson test (residuals autocorrelation).

Parameters
----------
vdf: vDataFrame
    input vDataFrame.
column: str
    Input vcolumn used as response.
ts: str
    vcolumn used as timeline. It will be to use to order the data. It can be
    a numerical or type date like (date, datetime, timestamp...) vcolumn.
X: list
    Input vcolumns used as predictors.
by: list, optional
    vcolumns used in the partition.

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
            ("X", X, [list],),
            ("by", by, [list],),
            ("vdf", vdf, [vDataFrame,],),
        ],
    )
    columns_check(X + [column] + [ts] + by, vdf)
    column = vdf_columns_names([column], vdf)[0]
    ts = vdf_columns_names([ts], vdf)[0]
    X = vdf_columns_names(X, vdf)
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
    query = "CREATE VIEW {} AS SELECT {}, {}, {}{} FROM {}".format(
        relation_name,
        ", ".join(X),
        column,
        ts,
        ", {}".format(", ".join(by)) if by else "",
        vdf.__genSQL__(),
    )
    vdf._VERTICAPY_VARIABLES_["cursor"].execute(query)
    model = LinearRegression(
        name, vdf._VERTICAPY_VARIABLES_["cursor"], solver="Newton", max_iter=1000
    )
    model.fit(relation_name, X, column)
    query = "(SELECT et, LAG(et) OVER({}ORDER BY {}) AS lag_et FROM (SELECT {}{}, {} - PREDICT_LINEAR_REG({} USING PARAMETERS model_name = '{}') AS et FROM {}) VERTICAPY_SUBTABLE) VERTICAPY_SUBTABLE".format(
        "PARTITION BY {} ".format(", ".join(by)) if (by) else "",
        ts,
        "{}, ".format(", ".join(by)) if by else "",
        ts,
        column,
        ", ".join(X),
        name,
        relation_name,
    )
    vdf.__executeSQL__(
        "SELECT SUM(POWER(et - lag_et, 2)) / SUM(POWER(et, 2)) FROM {}".format(query),
        title="Computes the Durbin Watson d.",
    )
    d = vdf._VERTICAPY_VARIABLES_["cursor"].fetchone()[0]
    vdf._VERTICAPY_VARIABLES_["cursor"].execute("DROP MODEL IF EXISTS {}".format(name))
    vdf._VERTICAPY_VARIABLES_["cursor"].execute(
        "DROP VIEW IF EXISTS {}".format(relation_name)
    )
    if d > 2.5 or d < 1.5:
        result = False
    else:
        result = True
    result = tablesample(
        {
            "index": ["Durbin Watson Index", "Residuals Stationarity"],
            "value": [d, result],
        }
    )
    return result


# ---#
def jarque_bera(vdf: vDataFrame, column: str, alpha: float = 0.05):
    """
---------------------------------------------------------------------------
Jarque Bera test (Distribution Normality).

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
    jb, n = vdf[column].agg(["jb", "count"]).values[column]
    pvalue = chi2.cdf(jb, n)
    result = True if pvalue < alpha else False
    result = tablesample(
        {
            "index": [
                "Jarque Bera Test Statistic",
                "p_value",
                "# Observations Used",
                "Distribution Normality",
            ],
            "value": [jb, pvalue, n, result],
        }
    )
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
    pvalue = norm.pdf(ZMK)
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
def plot_acf_pacf(
    vdf: vDataFrame, column: str, ts: str, by: list = [], p: (int, list) = 15,
):
    """
---------------------------------------------------------------------------
Draws the ACF and PACF Charts.

Parameters
----------
vdf: vDataFrame
    Input vDataFrame.
column: str
    Response column.
ts: str
    vcolumn used as timeline. It will be to use to order the data. 
    It can be a numerical or type date like (date, datetime, timestamp...) 
    vcolumn.
by: list, optional
    vcolumns used in the partition.
p: int/list, optional
    Int equals to the maximum number of lag to consider during the computation
    or List of the different lags to include during the computation.
    p must be positive or a list of positive integers.

Returns
-------
tablesample
    An object containing the result. For more information, see
    utilities.tablesample.
    """
    check_types(
        [
            ("column", column, [str],),
            ("ts", ts, [str],),
            ("by", by, [list],),
            ("p", p, [int, float],),
            ("vdf", vdf, [vDataFrame,],),
        ]
    )
    columns_check([column, ts] + by, vdf)
    by = vdf_columns_names(by, vdf)
    column, ts = vdf_columns_names([column, ts], vdf)
    acf = vdf.acf(ts=ts, column=column, by=by, p=p, show=False)
    pacf = vdf.pacf(ts=ts, column=column, by=by, p=p, show=False)
    result = tablesample(
        {
            "index": [i for i in range(0, len(acf.values["value"]))],
            "acf": acf.values["value"],
            "pacf": pacf.values["value"],
            "confidence": pacf.values["confidence"],
        },
    )
    fig = plt.figure(figsize=(10, 6)) if isnotebook() else plt.figure(figsize=(10, 6))
    plt.rcParams["axes.facecolor"] = "#FCFCFC"
    ax1 = fig.add_subplot(211)
    x, y, confidence = (
        result.values["index"],
        result.values["acf"],
        result.values["confidence"],
    )
    plt.xlim(-1, x[-1] + 1)
    ax1.bar(x, y, width=0.007 * len(x), color="#444444", zorder=1, linewidth=0)
    ax1.scatter(
        x, y, s=90, marker="o", facecolors="#FE5016", edgecolors="#FE5016", zorder=2
    )
    ax1.plot(
        [-1] + x + [x[-1] + 1],
        [0 for elem in range(len(x) + 2)],
        color="#FE5016",
        zorder=0,
    )
    ax1.fill_between(x, confidence, color="#FE5016", alpha=0.1)
    ax1.fill_between(x, [-elem for elem in confidence], color="#FE5016", alpha=0.1)
    ax1.set_title("Autocorrelation")
    y = result.values["pacf"]
    ax2 = fig.add_subplot(212)
    ax2.bar(x, y, width=0.007 * len(x), color="#444444", zorder=1, linewidth=0)
    ax2.scatter(
        x, y, s=90, marker="o", facecolors="#FE5016", edgecolors="#FE5016", zorder=2
    )
    ax2.plot(
        [-1] + x + [x[-1] + 1],
        [0 for elem in range(len(x) + 2)],
        color="#FE5016",
        zorder=0,
    )
    ax2.fill_between(x, confidence, color="#FE5016", alpha=0.1)
    ax2.fill_between(x, [-elem for elem in confidence], color="#FE5016", alpha=0.1)
    ax2.set_title("Partial Autocorrelation")
    plt.show()
    return result
