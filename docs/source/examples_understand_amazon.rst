.. _examples.understand.amazon:

Amazon
=======

This example uses the ``amazon`` dataset to predict the number of forest fires in Brazil. You can download a copy of the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/understand/amazon/amazon.ipynb>`_.

- **date:** Date of the record
- **number:** Number of forest fires
- **state:** State in Brazil

We'll follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem, and we'll do it without ever loading our data into memory.

Initialization
----------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp

    vp.__version__

Connect to Vertica. This example uses an existing connection called ``VerticaDSN``. 
For details on how to create a connection, see the :ref:`connection` tutorial.
You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Let's create a Virtual DataFrame of the dataset.

.. code-block:: python

    from verticapy.datasets import load_amazon

    amazon = load_amazon()
    amazon.head(5)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_amazon

    amazon = load_amazon()
    res = amazon.head(5)
    html_file = open("SPHINX_DIRECTORY/figures/examples_amazon_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_amazon_table_head.html

Data Exploration and Preparation
---------------------------------

We can explore our data by displaying descriptive statistics of all the columns.

.. code-block:: python

    amazon.describe(method = "categorical", unique = True)

.. ipython:: python
    :suppress:

    res = amazon.describe(method = "categorical", unique = True)
    html_file = open("SPHINX_DIRECTORY/figures/examples_amazon_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_amazon_table_describe.html

Using the :py:func:`~verticapy.vDataFrame.describe` method, we can see that our data ranges from the beginning of 1998 to the end of 2017.

.. code-block:: python

    amazon["date"].describe()

.. ipython:: python
    :suppress:

    res = amazon["date"].describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_amazon_table_describe_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_amazon_table_describe_2.html

Brazil has dry and rainy seasons. Knowing this, we would expect that the frequency of forest fires vary between seasons. Let's confirm our hypothesis using an autocorrelation plot with 48 lags (4 years).

.. code-block:: python

    amazon.acf(
        column = "number", 
        ts = "date",
        by = ["state"],
        p = 48,
    )

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = amazon.acf(
        column = "number", 
        ts = "date",
        by = ["state"],
        p = 48,
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_amazon_table_acf.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_amazon_table_acf.html

The process is not stationary. Let's use a Dickey-Fuller test to confirm our hypothesis.

.. code-block:: python

    from verticapy.machine_learning.model_selection.statistical_tests import adfuller

    adfuller(
        amazon,
        ts = "date", 
        column = "number", 
        by = ["state"], 
        p = 48,
    )

.. ipython:: python
    :suppress:

    from verticapy.machine_learning.model_selection.statistical_tests import adfuller

    res = adfuller(
        amazon,
        ts = "date", 
        column = "number", 
        by = ["state"], 
        p = 48,
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_amazon_adfuller.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_amazon_adfuller.html

The effects of each season seem pretty clear. We can see this graphically using the cumulative sum of the number of forest fires partitioned by states. If our hypothesis is correct, we should see staircase functions.

.. code-block:: python

    amazon.cumsum(
        "number", 
        by = ["state"], 
        order_by = ["date"], 
        name = "cum_sum",
    )
    amazon["cum_sum"].plot(
        ts = "date", 
        by = "state",
    )

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    amazon.cumsum(
        "number", 
        by = ["state"], 
        order_by = ["date"], 
        name = "cum_sum",
    )
    fig = amazon["cum_sum"].plot(
        ts = "date", 
        by = "state",
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_amazon_table_cum_sum.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_amazon_table_cum_sum.html

We can clearly observe the seasonality within each state, which contributes to an overall global seasonality. Let's plot the total number of forest fires to illustrate this more clearly.

.. code-block:: python

    import verticapy.sql.functions as fun

    amazon = amazon.groupby(
        ["date"], 
        [
            fun.sum(amazon["number"])._as("number"),
        ],
    )
    amazon["number"].plot(ts = "date")

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")

    import verticapy.sql.functions as fun

    amazon = amazon.groupby(
        ["date"], 
        [
            fun.sum(amazon["number"])._as("number"),
        ],
    )
    fig = amazon["number"].plot(ts = "date")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_amazon_table_plot_2.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_amazon_table_plot_2.html

Although it would be preferable to use seasonal decomposition and predict the residuals, let's build an ARIMA model on the data.

Machine Learning
-----------------

Since the seasonality occurs monthly, we set ``p = 12``. There is no trend in the data, and we observe some moving average in the residuals, so ``q`` should be around 2. Let's proceed with building the model.

.. code-block:: python

    from verticapy.machine_learning.vertica import ARIMA

    model = ARIMA(
        order = (12, 0, 2),
        missing = "drop",
    )
    model.fit(
        amazon,
        y = "number",
        ts = "date",
    )
    model.regression_report(start = 50)

.. ipython:: python
    :suppress:

    from verticapy.machine_learning.vertica import ARIMA

    model = ARIMA(
        order = (12, 0, 2),
        missing = "drop",
    )
    model.fit(
        amazon,
        y = "number",
        ts = "date",
    )
    res = model.regression_report(start = 50)
    html_file = open("SPHINX_DIRECTORY/figures/examples_amazon_table_ml_cv.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_amazon_table_ml_cv.html

Our model is quite good. Let's look at our predictions.

.. code-block:: python

    model.plot(
        vdf = amazon,
        ts = "date",
        y = "number",
        npredictions = 40,
        method = "auto",
    )

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = model.plot(
        vdf = amazon,
        ts = "date",
        y = "number",
        npredictions = 40,
        method = "auto",
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_amazon_table_plot_ml_2.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_amazon_table_plot_ml_2.html

The plot shows that our model has successfully captured the seasonality present in the data. However, to improve the model, we should remove the seasonality and focus on predicting the residuals directly. The current model is not entirely stable and requires further adjustments.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!