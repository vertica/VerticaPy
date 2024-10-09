.. _examples.learn.commodities:

Commodities
============

This example uses the 'Commodities' dataset to predict the price of different commodities. You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/learn/winequality/winequality.ipynb>`_.

- **date:** Date of the record.
- **Gold:** Price per ounce of Gold.
- **Oil:** Price per Barrel - West Texas Intermediate (WTI).
- **Spread:** Interest Rate Spreads.
- **Vix:** The CBOE Volatility Index (VIX) is a measure of expected price fluctuations in the SP500 Index options over the next 30 - days.
- **Dol_Eur:** How much $1 US is in euros.
- **SP500:** The S&P 500, or simply the S&P, is a stock market index that measures the stock performance of 500 large companies - listed on stock exchanges in the United States.

We will follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem.

Initialization
---------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp

    vp.__version__

Connect to Vertica. This example uses an existing connection called "VerticaDSN." 
For details on how to create a connection, see the :ref:`connection` tutorial.
You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Let's create a Virtual DataFrame of the dataset.

.. code-block:: python

    from verticapy.datasets import load_commodities

    commodities = load_commodities()
    commodities.head(100)

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.datasets import load_commodities
    commodities = load_commodities()
    res = commodities.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_commodities_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_head.html

Data Exploration and Preparation
---------------------------------

Let's explore the data by displaying descriptive statistics of all the columns.

.. code-block:: python

    commodities.describe(method = "all", unique = True)

.. ipython:: python
    :suppress:
    :okwarning:

    res = commodities.describe(method = "all", unique = True)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_commodities_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_describe.html

We have data from January 1986 to the beginning of August 2020. We don't have any missing values, so our data is already clean.

Let's draw the different variables.

.. code-block:: python

    commodities.plot(ts = "date")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = commodities.plot(ts = "date")
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_plot.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_plot.html

Some of the commodities have an upward monotonic trend and some others might be stationary. Let's use Augmented Dickey-Fuller tests to check our hypotheses.

.. code-block:: python

    from verticapy.machine_learning.model_selection.statistical_tests import adfuller
    from verticapy.core.tablesample import TableSample

    fuller = {}
    for commodity in ["Gold", "Oil", "Spread", "Vix", "Dol_Eur", "SP500"]:
        result = adfuller(
            commodities,
            column = commodity,
            ts = "date",
            p = 3,
            with_trend = True,
        )
        fuller["index"] = result["index"]
        fuller[commodity] = result["value"]
    fuller = TableSample(fuller)
    fuller

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.model_selection.statistical_tests import adfuller
    from verticapy.core.tablesample import TableSample

    fuller = {}
    for commodity in ["Gold", "Oil", "Spread", "Vix", "Dol_Eur", "SP500"]:
        result = adfuller(
            commodities,
            column = commodity,
            ts = "date",
            p = 3,
            with_trend = True,
        )
        fuller["index"] = result["index"]
        fuller[commodity] = result["value"]
    fuller = TableSample(fuller)
    res = fuller
    html_file = open("/project/data/VerticaPy/docs/figures/examples_commodities_table_adfuller.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_adfuller.html

As expected: The price of gold and the S&P 500 index are not stationary. Let's use the Mann-Kendall test to confirm the trends.

.. code-block:: python

    from verticapy.machine_learning.model_selection.statistical_tests import mkt

    kendall = {}
    for commodity in ["Gold", "SP500"]:
        result = mkt(
            commodities,
            column = commodity,
            ts = "date",
        )
        kendall["index"] = result["index"]
        kendall[commodity] = result["value"]
    kendall = TableSample(kendall)
    kendall

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.model_selection.statistical_tests import mkt

    kendall = {}
    for commodity in ["Gold", "SP500"]:
        result = mkt(
            commodities,
            column = commodity,
            ts = "date",
        )
        kendall["index"] = result["index"]
        kendall[commodity] = result["value"]
    kendall = TableSample(kendall)
    res = kendall
    html_file = open("/project/data/VerticaPy/docs/figures/examples_commodities_table_kendall.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_kendall.html

Our hypothesis is correct. We can also look at the correlation between the elapsed time and our variables to see the different trends.

.. code-block:: python

    import verticapy.sql.functions as fun

    commodities["elapsed_days"] = commodities["date"] - fun.min(commodities["date"])._over()
    commodities.corr(focus = "elapsed_days")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy.sql.functions as fun

    commodities["elapsed_days"] = commodities["date"] - fun.min(commodities["date"])._over()
    fig = commodities.corr(focus = "elapsed_days")
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_corr_1.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_corr_1.html

In the last plot, it's a bit hard to tell if 'Spread' is stationary. Let's draw it alone.

.. code-block:: python

    commodities["Spread"].plot(ts = "date")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = commodities["Spread"].plot(ts = "date")
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_plot_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_plot_2.html

We can see some sudden changes, so let's smooth the curve.

.. code-block:: python

    commodities.rolling(
        func = "avg",
        window = (-20, 0),
        columns = "Spread",
        order_by = ["date"],
        name = "Spread_smooth",
    )
    commodities["Spread_smooth"].plot(ts = "date")

.. ipython:: python
    :suppress:
    :okwarning:

    commodities.rolling(
        func = "avg",
        window = (-20, 0),
        columns = "Spread",
        order_by = ["date"],
        name = "Spread_smooth",
    )
    fig = commodities["Spread_smooth"].plot(ts = "date")
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_plot_3.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_plot_3.html

After each local minimum, there is a local maximum. Let's look at the number of lags needed to keep most of the information. To visualize this, we can draw the autocorrelation function (ACF) and partial autocorrelation function (PACF) plots.

.. code-block:: python

    commodities.acf(column = "Spread", ts = "date", p = 12)

.. ipython:: python
    :suppress:
    :okwarning:

    fig = commodities.acf(column = "Spread", ts = "date", p = 12)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_plot_acf_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_plot_acf_2.html

.. code-block:: python

    commodities.pacf(column = "Spread", ts = "date", p = 5)

.. ipython:: python
    :suppress:
    :okwarning:

    fig = commodities.pacf(column = "Spread", ts = "date", p = 5)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_plot_pacf_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_plot_pacf_2.html

We can clearly see the influence of the last two values on 'Spread', which makes sense. When the curve slightly changes its direction, it will increase/decrease until reaching a new local maximum/minimum. Only the recent values can help the prediction in case of autoregressive periodical model. The local minimums of interest rate spreads are indicators of an economic crisis.

We saw the correlation between the price-per-barrel of Oil and the time. Let's look at the time series plot of this variable.

.. code-block:: python

    commodities["Oil"].plot(ts = "date")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = commodities["Oil"].plot(ts = "date")
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_plot_4.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_plot_4.html

Moving on to the correlation matrix, we can see many events that changed drastically the values of commodities, and we know of a correlation between all of them. From here, we could look at how strong this correlation is, which will help us create a model that properly combines all the variable lags in its predictions.

.. code-block:: python

    commodities.corr(columns = ["Gold", "Oil", "Spread", "Vix", "Dol_Eur", "SP500"])

.. ipython:: python
    :suppress:
    :okwarning:

    fig = commodities.corr(columns = ["Gold", "Oil", "Spread", "Vix", "Dol_Eur", "SP500"])
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_corr_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_corr_2.html

We can see strong correlations between most of the variables. A vector autoregression (``VAR``) model seems ideal.

Machine Learning
-----------------

Let's create the ``VAR`` model to predict the value of various commodities.

.. code-block:: python

    from verticapy.machine_learning.vertica import VAR

    model = VAR(p = 5)
    model.fit(
        commodities,
        ts = "date",
        y = ["Gold", "Oil", "Spread", "Vix", "Dol_Eur", "SP500"],
    )
    model.score()

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import VAR

    model = VAR(p = 5)
    model.fit(
        commodities,
        ts = "date",
        y = ["Gold", "Oil", "Spread", "Vix", "Dol_Eur", "SP500"],
    )
    res = model.score()
    html_file = open("/project/data/VerticaPy/docs/figures/examples_commodities_table_ml_score.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_ml_score.html

Our model is excellent. Let's predict the values these commodities in the near future.

**Gold:**

.. code-block:: python

    model.plot(idx = 0, npredictions = 60)

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.plot(idx = 0, npredictions = 60)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_0.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_0.html

**Oil:**

.. code-block:: python

    model.plot(idx = 1, npredictions = 60)

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.plot(idx = 1, npredictions = 60)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_1.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_1.html

**Spread:**

.. code-block:: python

    model.plot(idx = 2, npredictions = 60)

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.plot(idx = 2, npredictions = 60)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_2.html

**Vix:**

.. code-block:: python

    model.plot(idx = 3, npredictions = 60)

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.plot(idx = 3, npredictions = 60)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_3.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_3.html

**Dol_Eur:**

.. code-block:: python

    model.plot(idx = 4, npredictions = 60)

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.plot(idx = 4, npredictions = 60)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_4.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_commodities_table_pred_plot_4.html

The model performs well but may be somewhat unstable. To improve it, we could apply data preparation techniques, such as seasonal decomposition, before building the ``VAR`` model.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!