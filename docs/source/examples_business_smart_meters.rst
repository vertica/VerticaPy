.. _examples.business.smart_meters:

Smart Meters
=============

This example uses the following datasets to predict peoples' electricity consumption. You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/business/smart_meters/smart_meters.ipynb>`_. We'll use the following datasets:

`sm_consumption <https://github.com/vertica/VerticaPy/blob/master/examples/business/smart_meters/sm_consumption.csv>`_

- **dateUTC:** Date and time of the record.
- **meterID:** Smart meter ID.
- **value:** Electricity consumed during 30 minute interval (in kWh).

`sm_weather <https://github.com/vertica/VerticaPy/blob/master/examples/business/smart_meters/sm_weather.csv>`_

- **dateUTC:** Date and time of the record.
- **temperature:** Temperature.
- **humidity:** Humidity.

`sm_meters <https://github.com/vertica/VerticaPy/blob/master/examples/business/smart_meters/sm_meters.csv>`_

- **longitude:** Longitude.
- **latitude:** Latitude.
- **residenceType:** 1 for Single-Family; 2 for Multi-Family; 3 for Appartement.

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

Create the :py:mod:`~verticapy.vDataFrame` of the datasets:

.. code-block:: python

    sm_consumption = vp.read_csv(
        "sm_consumption.csv",
        dtype = {
            "meterID": "Integer",
            "dateUTC": "Timestamp(6)",
            "value": "Float(22)",
        }
    )
    sm_weather = vp.read_csv(
        "sm_weather.csv",
        dtype = {
            "dateUTC": "Timestamp(6)",
            "temperature": "Float(22)",
            "humidity": "Float(22)",
        }
    )
    sm_meters = vp.read_csv("sm_meters.csv")

.. note:: You can let Vertica automatically decide the data type, or you can manually force the data type on any column as seen above.

.. code-block:: python

    sm_consumption.head(100)

.. ipython:: python
    :suppress:

    sm_consumption = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/smart_meters/sm_consumption.csv",
        dtype = {
            "meterID": "Integer",
            "dateUTC": "Timestamp(6)",
            "value": "Float(22)",
        }
    )
    sm_weather = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/smart_meters/sm_weather.csv",
        dtype = {
            "dateUTC": "Timestamp(6)",
            "temperature": "Float(22)",
            "humidity": "Float(22)",
        }
    )
    sm_meters = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/smart_meters/sm_meters.csv")
    res = sm_consumption.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_table_head.html

.. code-block:: python

    sm_weather.head(100)

.. ipython:: python
    :suppress:

    res = sm_weather.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_weather_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_weather_table_head.html

.. code-block:: python

    sm_meters.head(100)

.. ipython:: python
    :suppress:

    res = sm_weather.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_meters_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_meters_table_head.html

Data Exploration and Preparation
---------------------------------

Predicting energy consumption in households is very important. Surges in electricity use could cause serious power outages. In our case, we'll be using data on general household energy consumption in Ireland to predict consumption at various times.

In order to join the different data sources, we need to assume that the weather will be approximately the same across the entirety of Ireland. We'll use the date and time as the key to join 'sm_weather' and 'sm_consumption'.

Joining different datasets with interpolation
++++++++++++++++++++++++++++++++++++++++++++++

In VerticaPy, you can interpolate joins; Vertica will find the closest timestamp to the key and join the result.

.. code-block:: python

    sm_consumption_weather = sm_consumption.join(
        sm_weather,
        how = "left",
        on_interpolate = {"dateUTC": "dateUTC"},
        expr1 = ["dateUTC", "meterID", "value"],
        expr2 = ["humidity", "temperature"],
    )
    sm_consumption_weather.head(100)

.. ipython:: python
    :suppress:

    sm_consumption_weather = sm_consumption.join(
        sm_weather,
        how = "left",
        on_interpolate = {"dateUTC": "dateUTC"},
        expr1 = ["dateUTC", "meterID", "value"],
        expr2 = ["humidity", "temperature"],
    )
    res = sm_consumption_weather.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_table.html

Segmenting Latitude & Longitude using Clustering
+++++++++++++++++++++++++++++++++++++++++++++++++

The dataset 'sm_meters' is pretty important. In particular, the type of residence is probably a good predictor for electricity usage. We can create clusters of the different regions with k-means clustering based on longitude and latitude. Let's find the most suitable ``k`` using an elbow curve and scatter plot.

.. code-block:: python

    sm_meters.agg(["min", "max"])

.. ipython:: python
    :suppress:

    res = sm_meters.agg(["min", "max"])
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_meters_agg_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_meters_agg_table.html

.. ipython:: python
    :okwarning:

    from verticapy.machine_learning.model_selection import elbow
    from verticapy.datasets import load_world

    # Geo Plots are only available in Matplotlib.
    vp.set_option("plotting_lib", "matplotlib")

    # Loading the world map.
    world = load_world()

    # Plotting the final map.
    df = world.to_geopandas(geometry = "geometry")
    df = df[df["country"].isin(["Ireland", "United Kingdom"])]
    ax = df.plot(
        edgecolor = "black",
        color = "white",
        figsize = (10, 9),
    )

    @savefig examples_sm_meters_scatter.png
    sm_meters.scatter(["longitude", "latitude"], ax = ax)

.. image:: ../../docs/source/savefig/examples_sm_meters_scatter.png
    :width: 100%
    :align: center

Based on the scatter plot, five seems like the optimal number of clusters. Let's verify this hypothesis using an :py:func:`~verticapy.machine_learning.model_selection.elbow` curve.

.. code-block:: python

    # Switching back to Plotly.
    vp.set_option("plotting_lib", "plotly")

    elbow(sm_meters, ["longitude", "latitude"], n_cluster = (3, 8))

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = elbow(sm_meters, ["longitude", "latitude"], n_cluster = (3, 8))
    fig.write_html("SPHINX_DIRECTORY/figures/examples_sm_meters_elbow_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_meters_elbow_1.html

The elbow curve seems to confirm that five is the optimal number of clusters, so let's create a ``k-means`` model with that in mind.

.. ipython:: python

    from verticapy.machine_learning.vertica import KMeans

    model = KMeans(
        n_cluster = 5,
        init = [
            (-6.26980, 53.38127),
            (-9.06178, 53.25998),
            (-8.48641, 51.90216),
            (-7.12408, 52.24610),
            (-8.63985, 52.65945),
        ],
    )
    model.fit(
        sm_meters, 
        [
            "longitude",
             "latitude",
        ],
    )

Let's add our clusters to the :py:mod:`~verticapy.vDataFrame`.

.. ipython:: python

    sm_meters = model.predict(sm_meters, name = "region")

Let's draw a scatter plot of the different regions.

.. ipython:: python
    :okwarning:

    # Geo Plots are only available in Matplotlib.
    vp.set_option("plotting_lib", "matplotlib")

    ax = df.plot(
        edgecolor = "black",
        color = "white",
        figsize = (10, 9),
    )

    @savefig examples_sm_meters_scatter_2.png
    sm_meters.scatter(
        ["longitude", "latitude"], 
        by = "region",
        max_cardinality = 10,
        ax = ax,
    )

.. image:: ../../docs/source/savefig/examples_sm_meters_scatter_2.png
    :width: 100%
    :align: center

Dataset Enrichment
+++++++++++++++++++

Let's join ``sm_meters`` with ``sm_consumption_weather``.

.. code-block:: python

    sm_consumption_weather_region = sm_consumption_weather.join(
        sm_meters,
        how = "natural",
        expr1 = ["*"],
        expr2 = [
            "residenceType", 
            "region",
        ],
    )
    sm_consumption_weather_region.head(100)

.. ipython:: python
    :suppress:

    sm_consumption_weather_region = sm_consumption_weather.join(
        sm_meters,
        how = "natural",
        expr1 = ["*"],
        expr2 = [
            "residenceType", 
            "region",
        ],
    )
    res = sm_consumption_weather_region.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_region_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_region_table.html

Handling Missing Values
++++++++++++++++++++++++

Let's take care of our missing values.

.. code-block:: python

    sm_consumption_weather_region.count_percent()

.. ipython:: python
    :suppress:

    res = sm_consumption_weather_region.count_percent()
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_region_count_percent_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_region_count_percent_table.html

The variable 'value' has a few missing values that we can drop.

.. code-block:: python

    sm_consumption_weather_region["value"].dropna()
    sm_consumption_weather_region.count()

.. ipython:: python
    :suppress:

    sm_consumption_weather_region["value"].dropna()
    res = sm_consumption_weather_region.count()
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_region_count_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_region_count_2.html

Interpolation & Aggregations
+++++++++++++++++++++++++++++

Since power outages seem relatively common in each area, and the "value" represents the electricity consumed during 30 minute intervals (in kWh), it'd be a good idea to interpolate and aggregate the data to get a monthly average in electricity consumption per region.

Let's save our new dataset in the Vertica database.

.. ipython:: python

    vp.drop("sm_consumption_weather_region", method = "table")
    sm_consumption_weather_region.to_db(
        "sm_consumption_weather_region",
        relation_type = "table",
    )
    sm_consumption_weather_region_clean = vp.vDataFrame("sm_consumption_weather_region")

To get an equally-sliced dataset, we can then interpolate to fill any gaps. This operation is essential for creating correct time series models.

.. code-block:: python

    sm_consumption_weather_region_clean = sm_consumption_weather_region_clean.interpolate(
        ts = "dateUTC",
        rule = "30 minutes",
        method = {
            "value": "linear",
            "humidity": "linear",
            "temperature": "linear",
            "residenceType": "ffill",
            "region": "ffill",
        },
        by = ["meterID"],
    )
    sm_consumption_weather_region_clean.head(100)

.. ipython:: python
    :suppress:

    sm_consumption_weather_region_clean = sm_consumption_weather_region_clean.interpolate(
        ts = "dateUTC",
        rule = "30 minutes",
        method = {
            "value": "linear",
            "humidity": "linear",
            "temperature": "linear",
            "residenceType": "ffill",
            "region": "ffill",
        },
        by = ["meterID"],
    )
    res = sm_consumption_weather_region_clean.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_region_clean_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_weather_region_clean_1.html

Let's aggregate the data to figure out the monthly energy consumption for each smart meter. We can then save the result in the Vertica database.

.. code-block:: python

    import verticapy.sql.functions as fun

    sm_consumption_weather_region_clean["month"] = "MONTH(dateUTC)"
    sm_consumption_weather_region_clean["date_month"] = "DATE_TRUNC('MONTH', dateUTC::date)"
    sm_consumption_month = sm_consumption_weather_region_clean.groupby(
        columns = [
            "meterID",
            "region", 
            "residenceType",
            "month",
            "date_month",
        ],
        expr = [
            fun.sum(sm_consumption_weather_region["value"])._as("value"),
            fun.avg(sm_consumption_weather_region["temperature"])._as("avg_temperature"),
            fun.avg(sm_consumption_weather_region["humidity"])._as("avg_humidity"),
        ],
    ).filter(
        "date_month < '2015-09-01'",
    )
    vp.drop("sm_consumption_month", method = "table")
    sm_consumption_month.to_db(
        "sm_consumption_month",
        relation_type = "table",
        inplace = True,
    )

.. ipython:: python
    :suppress:

    import verticapy.sql.functions as fun

    sm_consumption_weather_region_clean["month"] = "MONTH(dateUTC)"
    sm_consumption_weather_region_clean["date_month"] = "DATE_TRUNC('MONTH', dateUTC::date)"
    sm_consumption_month = sm_consumption_weather_region_clean.groupby(
        columns = [
            "meterID",
            "region", 
            "residenceType",
            "month",
            "date_month",
        ],
        expr = [
            fun.sum(sm_consumption_weather_region["value"])._as("value"),
            fun.avg(sm_consumption_weather_region["temperature"])._as("avg_temperature"),
            fun.avg(sm_consumption_weather_region["humidity"])._as("avg_humidity"),
        ],
    ).filter(
        "date_month < '2015-09-01'",
    )
    vp.drop("sm_consumption_month", method = "table")
    res = sm_consumption_month.to_db(
        "sm_consumption_month",
        relation_type = "table",
        inplace = True,
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_month_clean_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_month_clean_2.html

Understanding the Data & Detecting Outliers
++++++++++++++++++++++++++++++++++++++++++++

Looking at three different smart meters, we can see a clear decrease in energy consumption during the summer followed by a sharp increase in the winter.

.. code-block:: python

    # Switching back to Plotly.
    vp.set_option("plotting_lib", "plotly")

    sm_consumption_month[sm_consumption_month["meterID"] == 10]["value"].plot(ts = "date_month")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = sm_consumption_month[sm_consumption_month["meterID"] == 10]["value"].plot(ts = "date_month")
    fig.write_html("SPHINX_DIRECTORY/figures/sm_consumption_month_plot_10.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/sm_consumption_month_plot_10.html

.. code-block:: python

    sm_consumption_month[sm_consumption_month["meterID"] == 12]["value"].plot(ts = "date_month")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = sm_consumption_month[sm_consumption_month["meterID"] == 12]["value"].plot(ts = "date_month")
    fig.write_html("SPHINX_DIRECTORY/figures/sm_consumption_month_plot_12.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/sm_consumption_month_plot_12.html

.. code-block:: python

    sm_consumption_month[sm_consumption_month["meterID"] == 14]["value"].plot(ts = "date_month")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = sm_consumption_month[sm_consumption_month["meterID"] == 14]["value"].plot(ts = "date_month")
    fig.write_html("SPHINX_DIRECTORY/figures/sm_consumption_month_plot_14.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/sm_consumption_month_plot_14.html

This behavior seems to be seasonal, but we don't have enough data to prove this.

Let's find outliers in the distribution by computing the ZSCORE per meterID.

.. code-block:: python

    std = fun.std(sm_consumption_month["value"])._over(by = [sm_consumption_month["meterID"]])
    avg = fun.avg(sm_consumption_month["value"])._over(by = [sm_consumption_month["meterID"]])
    sm_consumption_month["value_zscore"] = (sm_consumption_month["value"] - avg) / std
    sm_consumption_month.search("value_zscore > 4")

.. ipython:: python
    :suppress:
    :okwarning:

    std = fun.std(sm_consumption_month["value"])._over(by = [sm_consumption_month["meterID"]])
    avg = fun.avg(sm_consumption_month["value"])._over(by = [sm_consumption_month["meterID"]])
    sm_consumption_month["value_zscore"] = (sm_consumption_month["value"] - avg) / std
    res = sm_consumption_month.search("value_zscore > 4")
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_value_zscore_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_value_zscore_1.html

Four smart meters are outliers in energy consumption. We'll need to investigate to get more information.

.. code-block:: python

    sm_consumption_month[sm_consumption_month["meterID"] == 364]["value"].plot(ts = "date_month")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = sm_consumption_month[sm_consumption_month["meterID"] == 364]["value"].plot(ts = "date_month")
    fig.write_html("SPHINX_DIRECTORY/figures/sm_consumption_month_plot_1_364.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/sm_consumption_month_plot_1_364.html

.. code-block:: python

    sm_consumption_month[sm_consumption_month["meterID"] == 399]["value"].plot(ts = "date_month")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = sm_consumption_month[sm_consumption_month["meterID"] == 399]["value"].plot(ts = "date_month")
    fig.write_html("SPHINX_DIRECTORY/figures/sm_consumption_month_plot_1_399.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/sm_consumption_month_plot_1_399.html

.. code-block:: python

    sm_consumption_month[sm_consumption_month["meterID"] == 809]["value"].plot(ts = "date_month")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = sm_consumption_month[sm_consumption_month["meterID"] == 809]["value"].plot(ts = "date_month")
    fig.write_html("SPHINX_DIRECTORY/figures/sm_consumption_month_plot_1_809.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/sm_consumption_month_plot_1_809.html

.. code-block:: python

    sm_consumption_month[sm_consumption_month["meterID"] == 951]["value"].plot(ts = "date_month")

.. ipython:: python
    :suppress:
    :okwarning:

    fig = sm_consumption_month[sm_consumption_month["meterID"] == 951]["value"].plot(ts = "date_month")
    fig.write_html("SPHINX_DIRECTORY/figures/sm_consumption_month_plot_1_951.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/sm_consumption_month_plot_1_951.html

Data Encoding & Bivariate Analysis
+++++++++++++++++++++++++++++++++++

Since most of our data is categorical, let's encode them with One-hot encoding. We can then examine the correlations between the various categories.

.. code-block:: python

    sm_consumption_month = sm_consumption_month.one_hot_encode(
        ["region", "residenceType", "month"], 
        drop_first = False,
        max_cardinality = 20,
    )
    sm_consumption_month.head(100)

.. ipython:: python
    :suppress:

    sm_consumption_month = sm_consumption_month.one_hot_encode(
        ["region", "residenceType", "month"], 
        drop_first = False,
        max_cardinality = 20,
    )
    res = sm_consumption_month.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_month_clean_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_month_clean_4.html

Let's compute the Pearson correlation matrix.

.. code-block:: python

    sm_consumption_month.corr()

.. ipython:: python
    :suppress:

    fig = sm_consumption_month.corr(width = 820, with_numbers = False)
    fig.write_html("SPHINX_DIRECTORY/figures/examples_sm_consumption_month_corr_2.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_month_corr_2.html

There's a clear correlation between the month and energy consumption, but this isn't causal. Instead, we can think of the weather as having the direct influence on energy consumption. To accomodate for this view, we'll use the temperature as a predictor (rather than the month).

.. code-block:: python

    sm_consumption_month.corr(focus = "value")

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = sm_consumption_month.corr(focus = "value")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_sm_consumption_month_corr_3.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_month_corr_3.html

Global Behavior
++++++++++++++++

Let's look at this globally.

.. code-block:: python

    sm_consumption_final = sm_consumption_month.groupby(
        ["date_month"], 
        [
            fun.avg(sm_consumption_month["avg_temperature"])._as("avg_temperature"),
            fun.avg(sm_consumption_month["avg_humidity"])._as("avg_humidity"),
            fun.avg(sm_consumption_month["value"])._as("avg_value"),
        ],
    )
    sm_consumption_final.plot(ts = "date_month", columns = ["avg_value"])

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    sm_consumption_final = sm_consumption_month.groupby(
        ["date_month"], 
        [
            fun.avg(sm_consumption_month["avg_temperature"])._as("avg_temperature"),
            fun.avg(sm_consumption_month["avg_humidity"])._as("avg_humidity"),
            fun.avg(sm_consumption_month["value"])._as("avg_value"),
        ],
    )
    fig = sm_consumption_final.plot(ts = "date_month", columns = ["avg_value"])
    fig.write_html("SPHINX_DIRECTORY/figures/examples_sm_consumption_final_7.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_final_7.html

We expect to see a fall in energy consumption during summer and then an increase during the winter. A simple prediction could use the average value a year before.

.. code-block:: python

    sm_consumption_final["prediction"] = fun.case_when(
        sm_consumption_final["date_month"] < '2015-01-01', sm_consumption_final["avg_value"],
        fun.lag(sm_consumption_final["avg_value"], 12)._over(order_by = ["date_month"]),
    )
    sm_consumption_final.plot(ts = "date_month", columns = ["prediction", "avg_value"])

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    sm_consumption_final["prediction"] = fun.case_when(
        sm_consumption_final["date_month"] < '2015-01-01', sm_consumption_final["avg_value"],
        fun.lag(sm_consumption_final["avg_value"], 12)._over(order_by = ["date_month"]),
    )
    fig = sm_consumption_final.plot(ts = "date_month", columns = ["prediction", "avg_value"])
    fig.write_html("SPHINX_DIRECTORY/figures/examples_sm_consumption_final_8.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_final_8.html

.. ipython:: python

    sm_consumption_final.score("avg_value", "prediction", "r2")

As expected, our model's score is excellent.

Let's use machine learning to understand the influence of the weather and the humidity on energy consumption.

Machine Learning
-----------------

Let's create our model.

.. ipython:: python

    from verticapy.machine_learning.vertica import LinearRegression

    predictors = [
        "avg_temperature",
        "avg_humidity",
    ]
    model = LinearRegression(solver = "BFGS")
    model.fit(
        sm_consumption_final, 
        predictors,
        "avg_value",
    )

.. code-block:: python

    model.report("details")

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    res = model.report("details")
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_model_report_9.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_model_report_9.html

The model seems to be good with an adjusted R2 of 77.5%, and the F-Statistic indicates that at least one of the two predictors is useful. Let's look at the residual plot.

.. code-block:: python

    sm_consumption_final = model.predict(
        sm_consumption_final, 
        name = "value_prediction",
    )
    sm_consumption_final["residual"] = sm_consumption_final["avg_value"] - sm_consumption_final["value_prediction"]
    sm_consumption_final.scatter(["avg_value", "residual"])

.. ipython:: python
    :suppress:

    sm_consumption_final = model.predict(
        sm_consumption_final, 
        name = "value_prediction",
    )
    sm_consumption_final["residual"] = sm_consumption_final["avg_value"] - sm_consumption_final["value_prediction"]
    fig = sm_consumption_final.scatter(["avg_value", "residual"])
    fig.write_html("SPHINX_DIRECTORY/figures/examples_sm_consumption_final_1.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_final_1.html

Looking at the residual plot, we can see that the error variance varies by quite a bit. A possible suspect might be heteroscedasticity. Let's verify our hypothesis using a Breusch-Pagan test.

.. ipython:: python

    from verticapy.machine_learning.model_selection.statistical_tests import het_breuschpagan

    het_breuschpagan(sm_consumption_final, "residual", predictors)

The ``p-value`` is 4.81% and sits around the 5% threshold, so we can't really draw any conclusions.

Let's look at the entire regression report.

.. code-block:: python

    model.report()

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    res = model.report()
    html_file = open("SPHINX_DIRECTORY/figures/examples_sm_consumption_model_report_10.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()
    
.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_sm_consumption_model_report_10.html

Our model is very good; its median absolute error is around 13kWh.
With this model, we can make predictions about the energy consumption of households per region. If the usage exceeds what the model predicts, we can raise an alert and respond, for example, by regulating the electricity distributed to the region.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!