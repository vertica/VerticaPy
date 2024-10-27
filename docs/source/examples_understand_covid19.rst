.. _examples.understand.covid19:

COVID-19
=========

This example uses the ``covid19`` dataset to predict the number of deaths and cases one day in advance. You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/understand/covid19/covid19.ipynb>`_.

- **date:** Date of the record.
- **cases:** Number of people infected.
- **deaths:** Number of deaths.
- **state:** State.
- **fips:** The Federal Information Processing Standards (FIPS) code for the county.
- **county:** County.

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

Let's create a Virtual DataFrame of the dataset. The dataset is available `here <https://github.com/vertica/VerticaPy/blob/master/examples/understand/covid19/deaths.csv>`_.

.. code-block:: python

    from verticapy.datasets import load_commodities

    covid19 = vp.read_csv("deaths.csv")
    covid19.head(10)

.. ipython:: python
    :suppress:

    covid19 = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/covid19/deaths.csv")
    res = covid19.head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_commodities_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_head.html

Data Exploration and Preparation
---------------------------------

Let's explore the data by displaying descriptive statistics of all the columns.

.. code-block:: python

    covid19.describe(method = "categorical", unique = True)

.. ipython:: python
    :suppress:

    res = covid19.describe(method = "categorical", unique = True)
    html_file = open("SPHINX_DIRECTORY/figures/examples_covid19_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_describe.html

We have data from January 2020 to the beginning of May.

.. code-block:: python

    covid19["date"].describe()

.. ipython:: python
    :suppress:

    res = covid19["date"].describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_covid19_table_describe_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_describe_2.html

We'll try to predict the number of future deaths by using the statistics from previous days. We can drop the columns ``county`` and ``fips``, since the scope of our analysis is focused on the United States and the FIPS code isn't relevant to our predictions.

.. code-block:: python

    covid19.drop(["fips", "county"])

.. ipython:: python
    :suppress:

    res = covid19.drop(["fips", "county"])
    html_file = open("SPHINX_DIRECTORY/figures/examples_covid19_table_drop_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_drop_1.html

Let's sum the number of deaths and cases by state and date.

.. code-block:: python

    import verticapy.sql.functions as fun

    covid19 = covid19.groupby(
        [
            "state",
            "date",
        ],
        [
            fun.sum(covid19["deaths"])._as("deaths"),
            fun.sum(covid19["cases"])._as("cases"),
        ],
    )
    covid19.head(10)

.. ipython:: python
    :suppress:

    import verticapy.sql.functions as fun

    covid19 = covid19.groupby(
        [
            "state",
            "date",
        ],
        [
            fun.sum(covid19["deaths"])._as("deaths"),
            fun.sum(covid19["cases"])._as("cases"),
        ],
    )
    res = covid19.head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_covid19_table_clean_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_clean_1.html

Let's look at the autocorrelation graphic of the number of deaths.

.. code-block:: python

    covid19.acf(
        column = "deaths", 
        ts = "date",
        by = ["state"],
        p = 24,
    )

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = covid19.acf(
        column = "deaths", 
        ts = "date",
        by = ["state"],
        p = 24,
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_covid19_table_plot_acf.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_plot_acf.html

The process doesn't seem to be stationary. Let's use a Dickey-Fuller test to confirm our hypothesis.

.. code-block:: python

    from verticapy.machine_learning.model_selection.statistical_tests import adfuller

    adfuller(
        covid19,
        ts = "date", 
        column = "deaths", 
        by = ["state"], 
        p = 12,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.model_selection.statistical_tests import adfuller

    res = adfuller(
        covid19,
        ts = "date", 
        column = "deaths", 
        by = ["state"], 
        p = 12,
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_covid19_adfuller_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_adfuller_1.html

We can look at the cumulative number of deaths and its exponentiality.

.. code-block:: python

    covid19["deaths"].plot(
        ts = "date", 
        by = "state",
    )

.. ipython:: python
    :suppress:

    fig = covid19["deaths"].plot(
        ts = "date", 
        by = "state",
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_covid19_table_plot_3.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_plot_3.html

Let's plot this for the entire country.

.. code-block:: python

    covid = covid19.groupby(
        ["date"],
        [fun.sum(covid19["deaths"])._as("deaths")],
    )
    covid["deaths"].plot(ts = "date")

.. ipython:: python
    :suppress:

    covid = covid19.groupby(
        ["date"],
        [fun.sum(covid19["deaths"])._as("deaths")],
    )
    fig = covid["deaths"].plot(ts = "date")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_covid19_table_plot_4.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_plot_4.html

As you would expect, there's a clear correlation between the number of people infected and the number of deaths.

.. ipython:: python

    covid19.corr(["deaths", "cases"])

A vector autoregression (:py:mod:`~verticapy.machine_learning.vertica.tsa.VAR`) model can be very good to do the predictions. But first, let's encode the states to look at their influence.

.. code-block:: python

    covid19["state"].one_hot_encode()

.. ipython:: python
    :suppress:

    res = covid19["state"].one_hot_encode()
    html_file = open("SPHINX_DIRECTORY/figures/examples_covid19_one_hot_encode_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_one_hot_encode_1.html

Because of the upward monotonic trend, we can also look at the correlation between the days elapsed and the number of cases.

.. ipython:: python

    covid19["elapsed_days"] = covid19["date"] - fun.min(covid19["date"])._over(by = [covid19["state"]])

We can generate the SQL code of the :py:mod:`~verticapy.vDataFrame` 
to see what happens behind the scenes when we modify our data from within the :py:mod:`~verticapy.vDataFrame`.

.. ipython:: python

    print(covid19.current_relation())

The :py:mod:`~verticapy.vDataFrame` memorizes all of our operations on the data to dynamically generate the correct SQL statement and passes computation and aggregation to Vertica.

Let's see the correlation between the number of deaths and the other variables.

.. code-block:: python

    covid19.corr(focus = "deaths")

.. ipython:: python
    :suppress:

    fig = covid19.corr(focus = "deaths")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_covid19_table_plot_corr_5.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_plot_corr_5.html

We can see clearly a high correlation for some variables. We can use them to compute a ``SARIMAX`` model, but we'll stick to a :py:mod:`~verticapy.machine_learning.vertica.VAR` model for this study.

Let's compute the total number of deaths and cases to create our VAR model.

.. code-block:: python

    covid19 = vp.read_csv("deaths.csv").groupby(
        ["date"],
        [
            fun.sum(covid19["deaths"])._as("deaths"),
            fun.sum(covid19["cases"])._as("cases"),
        ],
    ).search("date > '04-01-2020'")

.. ipython:: python
    :suppress:

    covid19 = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/covid19/deaths.csv").groupby(
        ["date"],
        [
            fun.sum(covid19["deaths"])._as("deaths"),
            fun.sum(covid19["cases"])._as("cases"),
        ],
    ).search("date > '04-01-2020'")

Machine Learning
-----------------

Let's create a :py:mod:`~verticapy.machine_learning.vertica.VAR` model to predict the number of COVID-19 deaths and cases in the USA.

.. code-block:: python

    from verticapy.machine_learning.vertica.tsa import VAR

    model = VAR(p = 3)
    model.fit(
        covid19,
        ts = "date",
        y = ["cases", "deaths"],
        return_report = True,
    )
    model.score(start = 20)

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica.tsa import VAR

    model = VAR(p = 3)
    model.fit(
        covid19,
        ts = "date",
        y = ["cases", "deaths"],
        return_report = True,
    )
    res = model.score(start = 20)
    html_file = open("SPHINX_DIRECTORY/figures/examples_covid19_table_ml_score.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_ml_score.html

Our model is not bad. Let's predict the number of deaths in a near future.

Cases:
+++++++

.. code-block:: python

    model.plot(
        covid19,
        start = 37,
        npredictions = 10,
        idx = 0,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.plot(
        covid19,
        start = 37,
        npredictions = 10,
        idx = 0,
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_covid19_table_pred_plot_0.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_pred_plot_0.html

Deaths:
++++++++

.. code-block:: python

    model.plot(
        covid19,
        start = 37,
        npredictions = 10,
        idx = 1,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.plot(
        covid19,
        start = 37,
        npredictions = 10,
        idx = 1,
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_covid19_table_pred_plot_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_covid19_table_pred_plot_1.html

The model performs well but may be somewhat unstable. To improve it, we could apply data preparation techniques, such as seasonal decomposition, before building the VAR model.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!