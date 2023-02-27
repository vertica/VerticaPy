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
import datetime
from typing import Union
import numpy as np
from scipy.stats import chi2, norm, f

import verticapy._config.config as conf
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import schema_relation
from verticapy._utils._sql._sys import _executeSQL

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.linear_model import LinearRegression

from verticapy.sql.drop import drop


@save_verticapy_logs
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
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
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

    ts, column, by = vdf._format_colnames(ts, column, by)
    name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="linear_reg")
    relation_name = gen_tmp_name(
        schema=conf.get_option("temp_schema"), name="linear_reg_view"
    )
    drop(name, method="model")
    drop(relation_name, method="view")
    lag = [
        "LAG({}, 1) OVER ({}ORDER BY {}) AS lag1".format(
            column, "PARTITION BY {}".format(", ".join(by)) if (by) else "", ts
        )
    ]
    lag += [
        "LAG({}, {}) OVER ({}ORDER BY {}) - LAG({}, {}) OVER ({}ORDER BY {}) AS delta{}".format(
            column,
            i,
            f"PARTITION BY {', '.join(by)}" if (by) else "",
            ts,
            column,
            i + 1,
            f"PARTITION BY {', '.join(by)}" if (by) else "",
            ts,
            i,
        )
        for i in range(1, p + 1)
    ]
    lag += [
        "{} - LAG({}, 1) OVER ({}ORDER BY {}) AS delta".format(
            column, column, "PARTITION BY {}".format(", ".join(by)) if (by) else "", ts,
        )
    ]
    query = "CREATE VIEW {} AS SELECT /*+LABEL('statistical_tests.adfuller')*/ {}, {} AS ts FROM {}".format(
        relation_name,
        ", ".join(lag),
        "TIMESTAMPDIFF(SECOND, {}, MIN({}) OVER ())".format(ts, ts)
        if vdf[ts].isdate()
        else ts,
        vdf._genSQL(),
    )
    _executeSQL(query, print_time_sql=False)
    model = LinearRegression(name, solver="Newton", max_iter=1000)
    predictors = ["lag1"] + [f"delta{i}" for i in range(1, p + 1)]
    if with_trend:
        predictors += ["ts"]
    model.fit(relation_name, predictors, "delta")
    coef = model.get_attr("details")
    drop(name, method="model")
    drop(relation_name, method="view")
    if regresults:
        return coef
    coef = coef.transpose()
    DF = coef.values["lag1"][0] / (max(coef.values["lag1"][1], 1e-99))
    p_value = coef.values["lag1"][3]
    count = vdf.shape()[0]
    result = TableSample(
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


@save_verticapy_logs
def cochrane_orcutt(
    model,
    vdf: Union[vDataFrame, str],
    ts: str,
    prais_winsten: bool = False,
    drop_tmp_model: bool = True,
):
    """
Performs a Cochrane-Orcutt estimation.

Parameters
----------
model: vModel
    Linear regression object.
vdf: vDataFrame / str
    Input relation.
ts: str
    vcolumn of numeric or date-like type (date, datetime, timestamp, etc.)
    used as the timeline and to order the data.
prais_winsten: bool, optional
    If true, retains the first observation of the time series, increasing
    precision and efficiency. This configuration is called the 
    Prais–Winsten estimation.
drop_tmp_model: bool, optional
    If true, it drops the temporary model.

Returns
-------
model
    A Linear Model with the different information stored as attributes:
     - coef_        : Model's coefficients.
     - pho_         : Cochrane-Orcutt pho.
     - anova_table_ : ANOVA table.
     - r2_          : R2
    """
    if isinstance(vdf, str):
        vdf_tmp = vDataFrame(vdf)
    else:
        vdf_tmp = vdf.copy()
    ts = vdf._format_colnames(ts)
    name = gen_tmp_name(schema=schema_relation(model.model_name)[0], name="linear")
    param = model.get_params()
    model_tmp = type(model)(name)
    model_tmp.set_params(param)
    X, y = model.X, model.y
    print_info = conf.get_option("print_info")
    conf.set_option("print_info", False)
    if prais_winsten:
        vdf_tmp = vdf_tmp[X + [y, ts]].dropna()
    conf.set_option("print_info", print_info)
    prediction_name = gen_tmp_name(name="prediction")[1:-1]
    eps_name = gen_tmp_name(name="eps")[1:-1]
    model.predict(vdf_tmp, X=X, name=prediction_name)
    vdf_tmp[eps_name] = vdf_tmp[y] - vdf_tmp[prediction_name]
    query = "SELECT /*+LABEL('statistical_tests.cochrane_orcutt')*/ SUM(num) / SUM(den) FROM (SELECT {0} * LAG({0}) OVER (ORDER BY {1}) AS num,  POWER({0}, 2) AS den FROM {2}) x".format(
        eps_name, ts, vdf_tmp._genSQL()
    )
    pho = _executeSQL(
        query, title="Computing the Cochrane Orcutt pho.", method="fetchfirstelem"
    )
    for predictor in X + [y]:
        new_val = f"{predictor} - {pho} * LAG({predictor}) OVER (ORDER BY {ts})"
        if prais_winsten:
            new_val = f"COALESCE({new_val}, {predictor} * {(1 - pho ** 2) ** (0.5)})"
        vdf_tmp[predictor] = new_val
    model_tmp.drop()
    model_tmp.fit(vdf_tmp, X, y)
    model_tmp.pho_ = pho
    model_tmp.anova_table_ = model.regression_report("anova")
    model_tmp.r2_ = model.score("r2")
    if drop_tmp_model:
        model_tmp.drop()
    return model_tmp


@save_verticapy_logs
def durbin_watson(vdf: vDataFrame, eps: str, ts: str, by: list = []):
    """
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
    eps, ts, by = vdf._format_colnames(eps, ts, by)
    query = "(SELECT et, LAG(et) OVER({}ORDER BY {}) AS lag_et FROM (SELECT {} AS et, {}{} FROM {}) VERTICAPY_SUBTABLE) VERTICAPY_SUBTABLE".format(
        "PARTITION BY {} ".format(", ".join(by)) if (by) else "",
        ts,
        eps,
        ts,
        (", " + ", ".join(by)) if (by) else "",
        vdf._genSQL(),
    )
    d = _executeSQL(
        "SELECT /*+LABEL('statistical_tests.durbin_watson')*/ SUM(POWER(et - lag_et, 2)) / SUM(POWER(et, 2)) FROM {}".format(
            query
        ),
        title="Computing the Durbin Watson d.",
        method="fetchfirstelem",
    )
    return d


@save_verticapy_logs
def het_arch(vdf: vDataFrame, eps: str, ts: str, by: list = [], p: int = 1):
    """
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
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    eps, ts, by = vdf._format_colnames(eps, ts, by)
    X = []
    X_names = []
    for i in range(0, p + 1):
        X += [
            "LAG(POWER({}, 2), {}) OVER({}ORDER BY {}) AS lag_{}".format(
                eps, i, ("PARTITION BY " + ", ".join(by)) if (by) else "", ts, i
            )
        ]
        X_names += ["lag_{}".format(i)]
    query = "SELECT {} FROM {}".format(", ".join(X), vdf._genSQL())
    vdf_lags = vDataFrame(query)
    name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="linear_reg")
    model = LinearRegression(name)
    try:
        model.fit(vdf_lags, X_names[1:], X_names[0])
        R2 = model.score("r2")
    except:
        model.set_params({"solver": "bfgs"})
        model.fit(vdf_lags, X_names[1:], X_names[0])
        R2 = model.score("r2")
    finally:
        model.drop()
    n = vdf.shape()[0]
    k = len(X)
    LM = (n - p) * R2
    lm_pvalue = chi2.sf(LM, p)
    F = (n - 2 * p - 1) * R2 / (1 - R2) / p
    f_pvalue = f.sf(F, p, n - 2 * p - 1)
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
def ljungbox(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: list = [],
    p: int = 1,
    alpha: Union[int, float] = 0.05,
    box_pierce: bool = False,
):
    """
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
alpha: int / float, optional
    Significance Level. Probability to accept H0.
box_pierce: bool
    If set to True, the Box-Pierce statistic will be used.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    column, ts, by = vdf._format_colnames(column, ts, by)
    acf = vdf.acf(column=column, ts=ts, by=by, p=p, show=False)
    if p >= 2:
        acf = acf.values["value"][1:]
    else:
        acf = [acf]
    n = vdf[column].count()
    name = (
        "Ljung–Box Test Statistic" if not (box_pierce) else "Box-Pierce Test Statistic"
    )
    result = TableSample(
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


@save_verticapy_logs
def mkt(vdf: vDataFrame, column: str, ts: str, alpha: Union[int, float] = 0.05):
    """
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
alpha: int / float, optional
    Significance Level. Probability to accept H0.

Returns
-------
TableSample
    An object containing the result. For more information, see
    utilities.TableSample.
    """
    column, ts = vdf._format_colnames(column, ts)
    table = f"(SELECT {column}, {ts} FROM {vdf._genSQL()})"
    query = f"SELECT /*+LABEL('statistical_tests.mkt')*/ SUM(SIGN(y.{column} - x.{column})) FROM {table} x CROSS JOIN {table} y WHERE y.{ts} > x.{ts}"
    S = _executeSQL(
        query, title="Computing the Mann Kendall S.", method="fetchfirstelem"
    )
    try:
        S = float(S)
    except:
        S = None
    n = vdf[column].count()
    query = "SELECT /*+LABEL('statistical_tests.mkt')*/ SQRT(({0} * ({0} - 1) * (2 * {0} + 5) - SUM(row * (row - 1) * (2 * row + 5))) / 18) FROM (SELECT MAX(row) AS row FROM (SELECT ROW_NUMBER() OVER (PARTITION BY {1}) AS row FROM {2}) VERTICAPY_SUBTABLE GROUP BY row) VERTICAPY_SUBTABLE".format(
        n, column, vdf._genSQL()
    )
    STDS = _executeSQL(
        query,
        title="Computing the Mann Kendall S standard deviation.",
        method="fetchfirstelem",
    )
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
    result = TableSample(
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


@save_verticapy_logs
def seasonal_decompose(
    vdf: vDataFrame,
    column: str,
    ts: str,
    by: Union[str, list] = [],
    period: int = -1,
    polynomial_order: int = 1,
    estimate_seasonality: bool = True,
    rule: Union[str, datetime.timedelta] = None,
    mult: bool = False,
    two_sided: bool = False,
):
    """
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
by: str / list, optional
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
    assert period > 0 or polynomial_order > 0, ParameterError(
        "Parameters 'polynomial_order' and 'period' can not be both null."
    )
    ts, column, by = vdf._format_colnames(ts, column, by)
    if rule:
        vdf_tmp = vdf.interpolate(ts=ts, rule=period, method={column: "linear"}, by=by)
    else:
        vdf_tmp = vdf[[ts, column]]
    trend_name, seasonal_name, epsilon_name = (
        "{}_trend".format(column[1:-1]),
        "{}_seasonal".format(column[1:-1]),
        "{}_epsilon".format(column[1:-1]),
    )
    by, by_tmp = (
        "" if not (by) else "PARTITION BY " + ", ".join(by) + " ",
        by,
    )
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
        name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="linear_reg")
        model = LinearRegression(name=name, solver="bfgs", max_iter=100, tol=1e-6)
        model.drop()
        model.fit(vdf_poly, X, column)
        coefficients = [str(model.intercept_)] + [
            f"{model.coef_[i-1]} * POWER(ROW_NUMBER() OVER({by}ORDER BY {ts}), {i})"
            if i != 1
            else f"{model.coef_[0]} * ROW_NUMBER() OVER({by}ORDER BY {ts})"
            for i in range(1, polynomial_order + 1)
        ]
        vdf_tmp[trend_name] = " + ".join(coefficients)
        model.drop()
    if mult:
        vdf_tmp[seasonal_name] = f'{column} / NULLIFZERO("{trend_name}")'
    else:
        vdf_tmp[seasonal_name] = vdf_tmp[column] - vdf_tmp[trend_name]
    if period <= 0:
        acf = vdf_tmp.acf(
            column=seasonal_name, ts=ts, p=23, acf_type="heatmap", show=False
        )
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
        vdf_seasonality[
            "t_cos"
        ] = f"COS(2 * PI() * ROW_NUMBER() OVER ({by}ORDER BY {ts}) / {period})"
        vdf_seasonality[
            "t_sin"
        ] = f"SIN(2 * PI() * ROW_NUMBER() OVER ({by}ORDER BY {ts}) / {period})"
        X = ["t_cos", "t_sin"]
        name = gen_tmp_name(schema=conf.get_option("temp_schema"), name="linear_reg")
        model = LinearRegression(name=name, solver="bfgs", max_iter=100, tol=1e-6)
        model.drop()
        model.fit(vdf_seasonality, X, seasonal_name)
        vdf_tmp[
            seasonal_name
        ] = f"{model.intercept_} + {model.coef_[0]} * COS(2 * PI() * ROW_NUMBER() OVER ({by}ORDER BY {ts}) / {period}) + {model.coef_[1]} * SIN(2 * PI() * ROW_NUMBER() OVER ({by}ORDER BY {ts}) / {period})"
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
