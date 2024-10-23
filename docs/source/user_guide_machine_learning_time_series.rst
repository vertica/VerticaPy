.. _user_guide.machine_learning.time_series:

============
Time Series
============

Time series models are a type of regression on a dataset with a timestamp label.

The following example creates a time series model to predict the number of forest fires in Brazil with the 'Amazon' dataset.

.. code-block::

    from verticapy.datasets import load_amazon

    amazon = load_amazon().groupby("date", "SUM(number) AS number")
    amazon.head(100)

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.datasets import load_amazon
    amazon = load_amazon().groupby("date", "SUM(number) AS number")
    res = amazon.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_ml_table_ts_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_table_ts_1.html

The feature 'date' tells us that we should be working with a time series model. To do predictions on time series, we use previous values called 'lags'.

To help visualize the seasonality of forest fires, we'll draw some autocorrelation plots.

.. code-block:: python

    amazon.acf(
        ts = "date", 
        column = "number",
        p = 24,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = amazon.acf(
        ts = "date", 
        column = "number",
        p = 24,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_ml_plot_ts_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_plot_ts_2.html

.. code-block:: python

    amazon.pacf(
        ts = "date", 
        column = "number",
        p = 8,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = amazon.pacf(
        ts = "date", 
        column = "number",
        p = 8,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_ml_plot_ts_3.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_plot_ts_3.html

Forest fires follow a predictable, seasonal pattern, so it should be easy to predict future forest fires with past data.

VerticaPy offers several models, including a multiple time series model. For this example, let's use a :py:func:`~verticapy.machine_learning.vertica.ARIMA` model.

.. ipython:: python

    from verticapy.machine_learning.vertica import ARIMA

    model = ARIMA(order = (12, 0, 1))
    model.fit(
        amazon, 
        y = "number", 
        ts = "date",    
    )

Just like with other regression models, we'll evaluate our model with the :py:func:`~verticapy.machine_learning.vertica.ARIMA.report` method.

.. code-block::

    model.report(npredictions = 50, start = 50)

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.report(npredictions = 50, start = 50)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_ml_table_ts_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_table_ts_4.html

We can also draw our model using one-step ahead and dynamic forecasting.

.. code-block:: python

    model.plot(amazon, npredictions = 40,)

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = model.plot(amazon, npredictions = 40,)
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_ml_plot_ts_5.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_plot_ts_5.html

This concludes the fundamental lessons on machine learning algorithms in VerticaPy.