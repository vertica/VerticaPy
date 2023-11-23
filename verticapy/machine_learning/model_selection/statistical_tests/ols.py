"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
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
from typing import Literal, Optional, Union
import numpy as np
from scipy.stats import chi2, f

from vertica_python.errors import QueryError

import verticapy._config.config as conf
from verticapy._typing import NoneType, SQLColumns, SQLRelation
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import format_type
from verticapy._utils._gen import gen_tmp_name

from verticapy.core.tablesample.base import TableSample
from verticapy.core.vdataframe.base import vDataFrame

from verticapy.machine_learning.vertica.linear_model import LinearRegression

"""
OLS Tests: Heteroscedasticity.
"""


@save_verticapy_logs
def het_breuschpagan(
    input_relation: SQLRelation, eps: str, X: SQLColumns
) -> tuple[float, float, float, float]:
    """
    Uses the Breusch-Pagan to test a model for Heteroscedasticity.

    Parameters
    ----------
    input_relation: SQLRelation
        Input relation.
    eps: str
        Input residual vDataColumn.
    X: list
        The exogenous variables to test.

    Returns
    -------
    tuple
        Lagrange Multiplier statistic, LM pvalue,
        F statistic, F pvalue

    Examples
    ---------

    Initialization
    ^^^^^^^^^^^^^^^

    Let's try this test on a dummy dataset that has the
    following elements:

    - x (a predictor)
    - y (the response)
    - Random noise

    .. note::

        This metric requires ``eps``, which represents
        the difference between the predicted value
        and the true value. If you already have ``eps``
        available, you can directly use it instead of
        recomputing it, as demonstrated in the example
        below.

    Before we begin we can import the necessary libraries:

    .. ipython:: python

        import verticapy as vp
        import numpy as np
        from verticapy.machine_learning.vertica.linear_model import LinearRegression

    Example 1: Homoscedasticity
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Next, we can create some values with random
    noise:

    .. ipython:: python

        y_vals = [0, 2, 4, 6, 8, 10] + np.random.normal(0, 0.4, 6)

    We can use those values to create the :py:class:`vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "x": [0, 1, 2, 3, 4, 5],
                "y": y_vals,
            }
        )

    We can initialize a regression model:

    .. ipython:: python

        model = LinearRegression()

    Fit that model on the dataset:

    .. ipython:: python

        model.fit(input_relation = vdf, X = "x", y = "y")

    We can create a column in the :py:class:`vDataFrame` that
    has the predictions:

    .. ipython:: python

        model.predict(vdf, X = "x", name = "y_pred")

    Then we can calculate the residuals i.e. ``eps``:

    .. ipython:: python

        vdf["eps"] = vdf["y"] - vdf["y_pred"]

    We can plot the residuals to see the trend:

    .. code-block:: python

        vdf.scatter(["x", "eps"])

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = vdf.scatter(["x", "eps"], width = 550)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_breuschpagan.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_breuschpagan.html

    Notice the randomness of the residuals with respect to x.
    This shows that the noise is homoscedestic.

    To test its score, we can import the test function:

    .. ipython:: python

        from verticapy.machine_learning.model_selection.statistical_tests import het_breuschpagan

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        lm_statistic, lm_pvalue, f_statistic, f_pvalue = het_breuschpagan(vdf, eps = "eps", X = "x")

    .. ipython:: python

        print(lm_statistic, lm_pvalue, f_statistic, f_pvalue)

    As the noise was not heteroscedestic, we got higher
    p_value scores and lower statistics score.

    .. note::

        A ``p_value`` in statistics represents the
        probability of obtaining results as extreme
        as, or more extreme than, the observed data,
        assuming the null hypothesis is true.
        A *smaller* p-value typically suggests
        stronger evidence against the null hypothesis
        i.e. the test data does not have
        a heteroscedestic noise in the current case.

        However, *small* is a relative term. And
        the choice for the threshold value which
        determines a "small" should be made before
        analyzing the data.

        Generally a ``p-value`` less than 0.05
        is considered the threshold to reject the
        null hypothesis. But it is not always
        the case -
        `read more <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10232224/#:~:text=If%20the%20p%2Dvalue%20is,necessarily%20have%20to%20be%200.05.>`_

    .. note::

        F-statistics tests the overall significance
        of a model, while LM statistics tests the
        validity of linear restrictions on model
        parameters. High values indicate heterescedestic
        noise in this case.

    Example 2: Heteroscedasticity
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We can contrast the above result with a dataset that
    has **heteroscedestic noise** below:

    .. ipython:: python

        # y values
        y_vals = np.array([0, 2, 4, 6, 8, 10])

        # Adding some heteroscedestic noise
        y_vals = y_vals + [0.5, 0.3, 0.2, 0.1, 0.05, 0]

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "x": [0, 1, 2, 3, 4, 5],
                "y": y_vals,
            }
        )

    We can intialize a regression model:

    .. ipython:: python

        model = LinearRegression()

    Fit that model on the dataset:

    .. ipython:: python

        model.fit(input_relation = vdf, X = "x", y = "y")

    We can create a column in the :py:class:`vDataFrame` that
    has the predictions:

    .. ipython:: python

        model.predict(vdf, X = "x", name = "y_pred")

    Then we can calculate the residual i.e. ``eps``:

    .. ipython:: python

        vdf["eps"] = vdf["y"] - vdf["y_pred"]

    We can plot the residuals to see the trend:

    .. code-block:: python

        vdf.scatter(["x", "eps"])

    .. ipython:: python
        :suppress:

        fig = vdf.scatter(["x", "eps"], width = 550)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_breuschpagan_2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_breuschpagan_2.html

    Notice the relationship of the residuals with
    respect to x. This shows that the noise is
    heteroscedestic.

    Now we can perform the test on this dataset:

    .. ipython:: python

        lm_statistic, lm_pvalue, f_statistic, f_pvalue = het_breuschpagan(vdf, eps = "eps", X = "x")

    .. ipython:: python

        print(lm_statistic, lm_pvalue, f_statistic, f_pvalue)

    ..note::

        Notice the contrast of the two test results. In this
        dataset, the noise was heteroscedestic so we got very low
        p_value scores and higher statistics score. Thus confirming
        that the noise was in fact heteroscedestic.

        For more information check out
        `this link <https://www.statology.org/breusch-pagan-test/>`_.
    """
    if isinstance(input_relation, vDataFrame):
        vdf = input_relation.copy()
    else:
        vdf = vDataFrame(input_relation)
    X = format_type(X, dtype=list)
    eps, X = vdf.format_colnames(eps, X)
    model = LinearRegression()
    vdf_copy = vdf.copy()
    vdf_copy["v_eps2"] = vdf_copy[eps] ** 2
    try:
        model.fit(
            vdf_copy,
            X,
            "v_eps2",
            return_report=True,
        )
        R2 = model.score(metric="r2")
    except QueryError:
        model.set_params({"solver": "bfgs"})
        model.fit(
            vdf_copy,
            X,
            "v_eps2",
            return_report=True,
        )
        R2 = model.score(metric="r2")
    finally:
        model.drop()
    n = vdf.shape()[0]
    k = len(X)
    LM = n * R2
    lm_pvalue = chi2.sf(LM, k)
    F = (n - k - 1) * R2 / (1 - R2) / k
    f_pvalue = f.sf(F, k, n - k - 1)
    return LM, lm_pvalue, F, f_pvalue


@save_verticapy_logs
def het_goldfeldquandt(
    input_relation: SQLRelation,
    y: str,
    X: SQLColumns,
    idx: int = 0,
    split: float = 0.5,
    alternative: Literal["increasing", "decreasing", "two-sided"] = "increasing",
) -> tuple[float, float]:
    """
    Goldfeld-Quandt Homoscedasticity test.

    Parameters
    ----------
    input_relation: SQLRelation
        Input relation.
    y: str
        Response Column.
    X: SQLColumns
        Exogenous Variables.
    idx: int, optional
        Column index of variable according to which observations
        are sorted for the split.
    split: float, optional
        Float to indicate where to split (Example: 0.5 to split
        on the median).
    alternative: str, optional
        Specifies  the  alternative hypothesis  for  the  p-value
        calculation, one of the following variances: "increasing",
        "decreasing", "two-sided".

    Returns
    -------
    tuple
        statistic, p_value

    Examples
    ---------

    Initialization
    ^^^^^^^^^^^^^^^

    Let's try this test on a dummy dataset that has the
    following elements:

    - x (a predictor)
    - y (the response)
    - Random noise

    Before we begin we can import the necessary libraries:

    .. ipython:: python

        import verticapy as vp
        import numpy as np

    Example 1: Homoscedasticity
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Next, we can create some values with random
    noise:

    .. ipython:: python

        N = 50 # Number of rows
        x_val = list(range(N))
        y_val = [x * 2 for x in x_val] + np.random.normal(0, 0.4, N)

    We can use those values to create the :py:class:`vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "x": x_val,
                "y": y_val,
            }
        )


    We can plot the values to see the trend:

    .. code-block:: python

        vdf.scatter(["x", "y"])

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = vdf.scatter(["x", "y"], width = 550)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_goldfeldquandt.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_goldfeldquandt.html

    Notice the randomness of the data with respect to x.
    This shows that the noise is homoscedestic.

    To test its score, we can import the test function:

    .. ipython:: python

        from verticapy.machine_learning.model_selection.statistical_tests import het_goldfeldquandt

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        statistic, pvalue = het_goldfeldquandt(vdf, y = "y", X = "x")

    .. ipython:: python

        print(statistic, pvalue)

    As the noise was not heteroscedestic, we got higher
    p_value scores and lower statistics score.

    .. note::

        A ``p_value`` in statistics represents the
        probability of obtaining results as extreme
        as, or more extreme than, the observed data,
        assuming the null hypothesis is true.
        A *smaller* p-value typically suggests
        stronger evidence against the null hypothesis
        i.e. the test data does not have
        a heteroscedestic noise in the current case.

        However, *small* is a relative term. And
        the choice for the threshold value which
        determines a "small" should be made before
        analyzing the data.

        Generally a ``p-value`` less than 0.05
        is considered the threshold to reject the
        null hypothesis. But it is not always
        the case -
        `read more <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10232224/#:~:text=If%20the%20p%2Dvalue%20is,necessarily%20have%20to%20be%200.05.>`_

    .. note::

        F-statistics tests the overall significance
        of a model, while LM statistics tests the
        validity of linear restrictions on model
        parameters. High values indicate heterescedestic
        noise in this case.

    Example 2: Heteroscedasticity
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We can contrast the above result with a dataset that
    has **heteroscedestic noise** below:

    .. ipython:: python

        # y values
        x_val = list(range(N))
        y_val = [x * 2 for x in x_val]

        # Adding some heteroscedestic noise
        y_val = [x + np.random.normal() for x in y_val]

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "x": x_val,
                "y": y_val,
            }
        )


    We can plot the data to see the trend:

    .. code-block:: python

        vdf.scatter(["x", "y"])

    .. ipython:: python
        :suppress:

        fig = vdf.scatter(["x", "y"], width = 550)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_goldfeldquandt_2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_goldfeldquandt_2.html

    Notice the relationship of the residuals with
    respect to x. This shows that the noise is
    heteroscedestic.

    Now we can perform the test on this dataset:

    .. ipython:: python

        statistic, pvalue = het_goldfeldquandt(vdf, y = "y", X = "x")

    .. ipython:: python

        print(statistic, pvalue)

    ..note::

        Notice the contrast of the two test results. In this
        dataset, the noise was heteroscedestic so we got very low
        p_value scores and higher statistics score. Thus confirming
        that the noise was in fact heteroscedestic.
    """

    def model_fit(
        input_relation: SQLRelation, X: SQLColumns, y: str, model: LinearRegression
    ) -> LinearRegression:
        """
        helper functions used to fit the Linear Regression model.
        """
        mse = []
        for vdf_tmp in input_relation:
            model.drop()
            model.fit(
                vdf_tmp,
                X,
                y,
                return_report=True,
            )
            mse += [model.score(metric="mse")]
            model.drop()
        return mse

    if isinstance(input_relation, vDataFrame):
        vdf = input_relation.copy()
    else:
        vdf = vDataFrame(input_relation)
    X = format_type(X, dtype=list)
    y, X = vdf.format_colnames(y, X)
    split_value = vdf[X[idx]].quantile(split)
    vdf_0_half = vdf.search(vdf[X[idx]] < split_value)
    vdf_1_half = vdf.search(vdf[X[idx]] > split_value)
    model = LinearRegression()
    try:
        mse0, mse1 = model_fit([vdf_0_half, vdf_1_half], X, y, model)
    except QueryError:
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
    return F, f_pvalue


@save_verticapy_logs
def het_white(
    input_relation: SQLRelation, eps: str, X: SQLColumns
) -> tuple[float, float, float, float]:
    """
    White’s Lagrange Multiplier Test for Heteroscedasticity.

    Parameters
    ----------
    input_relation: SQLRelation
        Input relation.
    eps: str
        Input residual vDataColumn.
    X: str
        Exogenous Variables to test the Heteroscedasticity on.

    Returns
    -------
    tuple
        Lagrange Multiplier statistic, LM pvalue,
        F statistic, F pvalue

    Examples
    ---------

    Initialization
    ^^^^^^^^^^^^^^^

    Let's try this test on a dummy dataset that has the
    following elements:

    - x (a predictor)
    - y (the response)
    - Random noise

    .. note::

        This metric requires ``eps``, which represents
        the difference between the predicted value
        and the true value. If you already have ``eps``
        available, you can directly use it instead of
        recomputing it, as demonstrated in the example
        below.

    Before we begin we can import the necessary libraries:

    .. ipython:: python

        import verticapy as vp
        import numpy as np
        from verticapy.machine_learning.vertica.linear_model import LinearRegression

    Example 1: Homoscedasticity
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Next, we can create some values with random
    noise:

    .. ipython:: python

        y_vals = [0, 2, 4, 6, 8, 10] + np.random.normal(0, 0.4, 6)

    We can use those values to create the :py:class:`vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "x": [0, 1, 2, 3, 4, 5],
                "y": y_vals,
            }
        )

    We can initialize a regression model:

    .. ipython:: python

        model = LinearRegression()

    Fit that model on the dataset:

    .. ipython:: python

        model.fit(input_relation = vdf, X = "x", y = "y")

    We can create a column in the :py:class:`vDataFrame` that
    has the predictions:

    .. ipython:: python

        model.predict(vdf, X = "x", name = "y_pred")

    Then we can calculate the residuals i.e. ``eps``:

    .. ipython:: python

        vdf["eps"] = vdf["y"] - vdf["y_pred"]

    We can plot the residuals to see the trend:

    .. code-block:: python

        vdf.scatter(["x", "eps"])

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = vdf.scatter(["x", "eps"], width = 550)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_white.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_white.html

    Notice the randomness of the residuals with respect to x.
    This shows that the noise is homoscedestic.

    To test its score, we can import the test function:

    .. ipython:: python

        from verticapy.machine_learning.model_selection.statistical_tests import het_white

    And simply apply it on the :py:class:`vDataFrame`:

    .. ipython:: python

        lm_statistic, lm_pvalue, f_statistic, f_pvalue = het_white(vdf, eps = "eps", X = "x")

    .. ipython:: python

        print(lm_statistic, lm_pvalue, f_statistic, f_pvalue)

    As the noise was not heteroscedestic, we got higher
    p_value scores and lower statistics score.

    .. note::

        A ``p_value`` in statistics represents the
        probability of obtaining results as extreme
        as, or more extreme than, the observed data,
        assuming the null hypothesis is true.
        A *smaller* p-value typically suggests
        stronger evidence against the null hypothesis
        i.e. the test data does not have
        a heteroscedestic noise in the current case.

        However, *small* is a relative term. And
        the choice for the threshold value which
        determines a "small" should be made before
        analyzing the data.

        Generally a ``p-value`` less than 0.05
        is considered the threshold to reject the
        null hypothesis. But it is not always
        the case -
        `read more <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10232224/#:~:text=If%20the%20p%2Dvalue%20is,necessarily%20have%20to%20be%200.05.>`_

    .. note::

        F-statistics tests the overall significance
        of a model, while LM statistics tests the
        validity of linear restrictions on model
        parameters. High values indicate heterescedestic
        noise in this case.

    Example 2: Heteroscedasticity
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We can contrast the above result with a dataset that
    has **heteroscedestic noise** below:

    .. ipython:: python

        # y values
        y_vals = np.array([0, 2, 4, 6, 8, 10])

        # Adding some heteroscedestic noise
        y_vals = y_vals + [0.5, 0.3, 0.2, 0.1, 0.05, 0]

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "x": [0, 1, 2, 3, 4, 5],
                "y": y_vals,
            }
        )

    We can intialize a regression model:

    .. ipython:: python

        model = LinearRegression()

    Fit that model on the dataset:

    .. ipython:: python

        model.fit(input_relation = vdf, X = "x", y = "y")

    We can create a column in the :py:class:`vDataFrame` that
    has the predictions:

    .. ipython:: python

        model.predict(vdf, X = "x", name = "y_pred")

    Then we can calculate the residual i.e. ``eps``:

    .. ipython:: python

        vdf["eps"] = vdf["y"] - vdf["y_pred"]

    We can plot the residuals to see the trend:

    .. code-block:: python

        vdf.scatter(["x", "eps"])

    .. ipython:: python
        :suppress:

        fig = vdf.scatter(["x", "eps"], width = 550)
        fig.write_html("SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_white_2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_machine_learning_model_selection_ols_het_white_2.html

    Notice the relationship of the residuals with
    respect to x. This shows that the noise is
    heteroscedestic.

    Now we can perform the test on this dataset:

    .. ipython:: python

        lm_statistic, lm_pvalue, f_statistic, f_pvalue = het_white(vdf, eps = "eps", X = "x")

    .. ipython:: python

        print(lm_statistic, lm_pvalue, f_statistic, f_pvalue)

    ..note::

        Notice the contrast of the two test results. In this
        dataset, the noise was heteroscedestic so we got very low
        p_value scores and higher statistics score. Thus confirming
        that the noise was in fact heteroscedestic.
    """
    if isinstance(input_relation, vDataFrame):
        vdf = input_relation.copy()
    else:
        vdf = vDataFrame(input_relation)
    X = format_type(X, dtype=list)
    eps, X = vdf.format_colnames(eps, X)
    X_0 = ["1"] + X
    variables = []
    variables_names = []
    for i in range(len(X_0)):
        for j in range(i, len(X_0)):
            if i != 0 or j != 0:
                variables += [f"{X_0[i]} * {X_0[j]} AS var_{i}_{j}"]
                variables_names += [f"var_{i}_{j}"]
    query = f"""
        SELECT 
            {', '.join(variables)}, 
            POWER({eps}, 2) AS v_eps2 
        FROM {vdf}"""
    vdf_white = vDataFrame(query)
    model = LinearRegression()
    try:
        model.fit(
            vdf_white,
            variables_names,
            "v_eps2",
            return_report=True,
        )
        R2 = model.score(metric="r2")
    except QueryError:
        model.set_params({"solver": "bfgs"})
        model.fit(
            vdf_white,
            variables_names,
            "v_eps2",
            return_report=True,
        )
        R2 = model.score(metric="r2")
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
    return LM, lm_pvalue, F, f_pvalue


"""
OLS Tests: Multicollinearity.
"""


@save_verticapy_logs
def variance_inflation_factor(
    input_relation: SQLRelation, X: SQLColumns, X_idx: Optional[int] = None
) -> Union[float, TableSample]:
    """
    Computes the variance inflation factor (VIF). It can be
    used  to detect multicollinearity in an OLS  Regression
    Analysis.

    Parameters
    ----------
    input_relation: SQLRelation
        Input relation.
    X: list
        Input Variables.
    X_idx: int
        Index of the exogenous variable in X. If empty,
        a TableSample is returned with all the  variables
        VIF.

    Returns
    -------
    float / TableSample
        VIF.

    Examples
    ---------

    Initialization
    ^^^^^^^^^^^^^^^

    Let's try this test on a dummy dataset that has the
    following elements:

    - data with multiple columns

    Before we begin we can import the necessary libraries:

    .. ipython:: python

        import verticapy as vp
        import numpy as np

    Next, we can create some exogenous columns
    with varying collinearity:

    .. ipython:: python
        :suppress:

        N = 50
        x_val_1 = list(range(N))
        x_val_2 = [2 * x + np.random.normal(scale = 4) for x in x_val_1]
        x_val_3 = np.random.normal(0, 4, N)

    .. code-block:: python

        N = 50
        x_val_1 = list(range(N))
        x_val_2 = [2 * x + np.random.normal(scale = 4) for x in x_val_1]
        x_val_3 = np.random.normal(0, 4, N)

    We can use those values to create the :py:class:`vDataFrame`:

    .. ipython:: python

        vdf = vp.vDataFrame(
            {
                "x1": x_val_1,
                "x2": x_val_2,
                "x3": x_val_3,
            }
        )

    Data Visualization
    ^^^^^^^^^^^^^^^^^^^

    We can plot the data to see any underlying collinearity:

    Let us first draw ``x1`` with ``x2``:

    .. code-block:: python

        vdf.scatter(["x1", "x2"])

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = vdf.scatter(["x1", "x2"], width = 550)
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_model_selection_statistical_tests_vif_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_model_selection_statistical_tests_vif_1.html

    We can see that ``x1`` and ``x2`` are very correlated.

    Next let us observe ``x1`` and ``x3``:

    .. code-block:: python

        vdf.scatter(["x1", "x3"])

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = vdf.scatter(["x1", "x3"], width = 550)
        fig.write_html("SPHINX_DIRECTORY/figures/machine_learning_model_selection_statistical_tests_vif_2.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_model_selection_statistical_tests_vif_2.html

    We can see that the two are not correlated.

    Now we can confirm our observations by carrying out the
    VIC test. First, we can import the test:

    .. ipython:: python

        from verticapy.machine_learning.model_selection.statistical_tests import variance_inflation_factor

    And then apply it on the exogenous columns:

    .. code-block:: python

        variance_inflation_factor(vdf, X = ["x1", "x2", "x3"])

    .. ipython:: python
        :suppress:

        result = variance_inflation_factor(vdf, X =["x1", "x2", "x3"])
        html_file = open("SPHINX_DIRECTORY/figures/machine_learning_model_selection_statistical_tests_vic_3.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/machine_learning_model_selection_statistical_tests_vic_3.html

    .. note::

        We can clearly see that ``x1`` and ``x2`` are
        correlated because of the high value of VIC.
        But there is no correlation with ``x3`` as
        the VIC value is close to 1.
    """
    if isinstance(input_relation, vDataFrame):
        vdf = input_relation.copy()
    else:
        vdf = vDataFrame(input_relation)
    X = format_type(X, dtype=list)
    X, X_idx = vdf.format_colnames(X, X_idx)
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
        model = LinearRegression()
        try:
            model.fit(
                vdf,
                X_r,
                y_r,
                return_report=True,
            )
            R2 = model.score(metric="r2")
        except QueryError:
            model.set_params({"solver": "bfgs"})
            model.fit(
                vdf,
                X_r,
                y_r,
                return_report=True,
            )
            R2 = model.score(metric="r2")
        finally:
            model.drop()
        if 1 - R2 != 0:
            return 1 / (1 - R2)
        else:
            return np.inf
    elif isinstance(X_idx, NoneType):
        VIF = []
        for i in range(len(X)):
            VIF += [variance_inflation_factor(vdf, X, i)]
        return TableSample({"X_idx": X, "VIF": VIF})
    else:
        raise IndexError(
            "Wrong type for Parameter X_idx.\n"
            f"Expected integer, found {type(X_idx)}."
        )
