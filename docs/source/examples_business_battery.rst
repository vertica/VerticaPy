.. _examples.business.battery:

Estimating Lithium-ion Battery Health
======================================

Introduction 
-------------

Lithium-based batteries - their cycles characteristics and aging
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Lithium-ion (or Li-ion) batteries are rechargeable batteries used for a variety of electronic devices, which range from eletric vehicles, smartphones, and even satellites.

However, despite their wide adoption, research isn't mature enough to avoid problems with battery health and safety, and given the ubiquity of consumer electronics using the technology, this has led to some poor outcomes that range from poor user-experience to public safety concerns (see, for example, the Samsung Galaxy Note 7 explosions from 2016).

Dataset
++++++++

In this example of **predictive maintenance**, we propose a data-driven method to estimate the health of a battery using the `Li-ion battery dataset <https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/>`_ released by NASA.

This dataset includes information on Li-ion batteries over several charge and discharge cycles at room temperature. Charging was at a constant current (CC) at 1.5A until the battery voltage reached 4.2V and then continued in a constant voltage (CV) mode until the charge current dropped to 20mA. 

Discharge was at a constant current (CC) level of 2A until the battery voltage fell to 2.7V.

You can download the Jupyter Notebook of the study 
`here <https://github.com/vertica/VerticaPy/blob/master/examples/business/battery/battery.ipynb>`_.

The dataset includes the following:

- **Voltage_measured:** Battery's terminal voltage (Volts) for charging and discharging cycles.
- **Current_measured:** Battery's output current (Amps) for charging and discharging cycles.
- **Temperature_measured:** Battery temperature (degree Celsius).
- **Current_charge:** Current measured at charger for charging cycles and at load for discharging cycles (Amps).
- **Voltage_charge:** Voltage measured at charger for charging cycles and at load for discharging ones (Volts).
- **Start_time:** Starting time of the cycle.
- **Time:** Time in seconds after the starting time for the cycle (seconds).
- **Capacity:** Battery capacity (Ahr) for discharging until 2.7V. Battery capacity is the product of the current drawn from the battery (while the battery is able to supply the load) until its voltage drops lower than a certain value for each cell.

We will follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem.

Initialization
----------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp
    vp.__version__

Connect to Vertica. This example uses an existing connection called "VerticaDSN." 
For details on how to create a connection, see the :ref:`connection` tutorial.
You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Before we import the data, we'll drop any existing schemas of the same name.

.. ipython:: python

    vp.drop("battery_data", method="schema")
    vp.create_schema("battery_data", True)

Let us now ingest the data.

.. code-block:: python

    battery5 = vp.read_csv("data/data.csv")

.. warning::
    
    This example uses a sample dataset. For the full analysis, you should consider using the complete dataset.


Understanding the Data
-----------------------

Let's examine our data. Here, we use :py:func:`~verticapy.vDataFrame.head` to retrieve the first five rows of the dataset.

.. ipython:: python
    :suppress:

    battery5 = vp.read_csv("SPHINX_DIRECTORY/source/_static/website/examples/data/battery/data.csv",)
    res = battery5
    html_file = open("SPHINX_DIRECTORY/figures/examples_battery_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_table_head.html

Let's perform a few aggregations with :py:func:`~verticapy.vDataFrame.describe` to get a high-level overview of the dataset.

.. code-block:: python

    battery5.describe()

.. ipython:: python
    :suppress:

    res = battery5.describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_battery_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_table_describe.html

To get a better idea of the changes between each cycle, we look at an aggregation at their start time, duration, and voltage at the beginning and the end of each cycle.

.. code-block:: python

    battery5["start_time"].describe()

.. ipython:: python
    :suppress:

    res = battery5["start_time"].describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_battery__start_time_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery__start_time_table_describe.html

To see how the voltage changes during the cycle, we extract the initial and final voltage measurements for each cycle.

.. code-block:: python

    battery5.analytic(
        func = "first_value",
        columns = "Voltage_measured",
        by = "start_time",
        order_by = {"Time": "asc"},
        name = "first_voltage_measured",
    )
    battery5.analytic(
        func = "first_value",
        columns = "Voltage_measured",
        by = "start_time",
        order_by = {"Time": "desc"},
        name = "last_voltage_measured",
    )
    cycling_info = battery5.groupby(
            columns = [
                "start_time",
                "type",
                "first_voltage_measured",
                "last_voltage_measured",
            ], 
            expr = [
                "COUNT(*) AS nr_of_measurements",
                "MAX(Time) AS cycle_duration",
            ],
    ).sort("start_time")
    cycling_info["cycle_id"] = "ROW_NUMBER() OVER(ORDER BY start_time)"
    cycling_info.head(100)

.. ipython:: python
    :suppress:

    battery5.analytic(
        func = "first_value",
        columns = "Voltage_measured",
        by = "start_time",
        order_by = {"Time": "asc"},
        name = "first_voltage_measured",
    )
    battery5.analytic(
        func = "first_value",
        columns = "Voltage_measured",
        by = "start_time",
        order_by = {"Time": "desc"},
        name = "last_voltage_measured",
    )
    cycling_info = battery5.groupby(
            columns = [
                "start_time",
                "type",
                "first_voltage_measured",
                "last_voltage_measured",
            ], 
            expr = [
                "COUNT(*) AS nr_of_measurements",
                "MAX(Time) AS cycle_duration",
            ],
    ).sort("start_time")
    cycling_info["cycle_id"] = "ROW_NUMBER() OVER(ORDER BY start_time)"
    res = cycling_info.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_battery_cycling_info.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_cycling_info.html

We can see from the "duration" column that charging seems to take a longer time than discharging.

Let's visualize this trend with an animated graph.

.. code-block:: python

    cycling_info.animated_bar(
        ts = "start_time",
        columns = ["type", "cycle_duration"],
    )

.. ipython:: python
    :suppress:
    :okwarning:

    import warnings
    warnings.filterwarnings("ignore")
    res = cycling_info.animated_bar(ts = "start_time",columns = ["type", "cycle_duration"])
    html_file = open("SPHINX_DIRECTORY/figures/examples_battery_animated_bar.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_animated_bar.html

The animated graph below shows how the cycles change throughout time. Another way we can verify that charging cycles are longer than discharging cycles is by looking at the average duration of each type of cycle.

.. code-block:: python

    cycling_info.bar(
        ["type"], 
        method = "avg", 
        of = "cycle_duration",
    )

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = cycling_info.bar(["type"], method = "avg", of = "cycle_duration")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_battery_bar_type.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_bar_type.html

In general, charging cycles are longer than discharging cycles.
 
Let's examine how voltage changes between cycles and their transitions.

.. code-block:: python

    cycling_info = cycling_info.groupby(
        "type",
        [
            "MIN(first_voltage_measured) AS min_first_voltage",
            "AVG(first_voltage_measured) AS avg_first_voltage",
            "MAX(first_voltage_measured) AS max_first_voltage",
            "MIN(last_voltage_measured)  AS min_last_voltage",
            "AVG(last_voltage_measured)  AS avg_last_voltage",
            "MAX(last_voltage_measured)  AS max_last_voltage",
        ],
    )
    cycling_info.head(100)

.. ipython:: python
    :suppress:
    :okwarning:

    cycling_info.groupby(
        "type",
        [
            "MIN(first_voltage_measured) AS min_first_voltage",
            "AVG(first_voltage_measured) AS avg_first_voltage",
            "MAX(first_voltage_measured) AS max_first_voltage",
            "MIN(last_voltage_measured)  AS min_last_voltage",
            "AVG(last_voltage_measured)  AS avg_last_voltage",
            "MAX(last_voltage_measured)  AS max_last_voltage",
        ],
    )
    res = cycling_info.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/examples_battery_cycling_info_after_groupby.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_cycling_info_after_groupby.html

From this table, it looks like batteries are charged until they are almost full (4.2V) and discharging doesn't begin until they are fully charged.

End-of-life (EOL) criteria for batteries is usually defined as when the battery capacity is lower than 70%-80% of its rated capacity. Since the rated capacity by the manufacturer for this battery is 2Ah, this battery is considered EOL when its capacity reaches 2Ah x 70% = 1.4Ah.

Let's plot the capacity curve of the battery with its smoothed version and observe when it reaches the degradation criteria. 

But first we need to perform some preprocessing.

.. code-block:: python

    discharging_data = battery5[battery5["type"] == "discharge"]
    d_cap = discharging_data[["start_time", "Capacity"]].groupby(["start_time", "Capacity"])
    d_cap["discharge_id"] = "ROW_NUMBER() OVER(ORDER BY start_time, Capacity)"
    d_cap.rolling(
        func = "mean",
        columns = "capacity",
        window = (-100, -1),
        name = "smooth_capacity",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    discharging_data = battery5[battery5["type"] == "discharge"]
    d_cap = discharging_data[["start_time", "Capacity"]].groupby(["start_time", "Capacity"])
    d_cap["discharge_id"] = "ROW_NUMBER() OVER(ORDER BY start_time, Capacity)"
    res = d_cap.rolling(
        func = "mean",
        columns = "capacity",
        window = (-100, -1),
        name = "smooth_capacity",
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_battery_cycling_info_after_rollign_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_cycling_info_after_rollign_2.html

Now we can plot the graphs. In VerticaPy we have multiple options to plot the graphs with different syntax of customization. For a complete list of all the graphs and their options check out the :ref:`chart_gallery`.

Now let's first try to plot this using Matplotlib:

.. code-block:: python

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import axhline

    # Switch the plotting library to Matplotlib
    vp.set_option("plotting_lib", "matplotlib")

    fig = plt.figure()
    ax = d_cap.plot(ts = "discharge_id", columns = ["Capacity", "smooth_capacity"])
    ax.axhline(y = 1.4, label = "End-of-life criteria")
    ax.set_title("Capacity degradation curve of the battery, its smoothed version and its end-of-life threshold")
    ax.legend() 
    plt.show()

.. ipython:: python

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import axhline

    # Switch the plotting library to Matplotlib
    vp.set_option("plotting_lib", "matplotlib")

    fig = plt.figure()
    ax = d_cap.plot(ts = "discharge_id", columns = ["Capacity", "smooth_capacity"])
    ax.axhline(y = 1.4, label = "End-of-life criteria")
    ax.set_title("Capacity degradation curve of the battery, its smoothed version and its end-of-life threshold")
    ax.legend()
    @savefig examples_battery_matplotlib_capacity_degradation.png 
    plt.show()

We can now try to plot it using Plotly. We can conveniently switch between the plotting libraries using:

.. ipython:: python

    # Switch the plotting library to Plotly
    vp.set_option("plotting_lib", "plotly")

.. code-block:: python

    import plotly.graph_objects as go

    plot = d_cap.plot(ts = "discharge_id", columns = ["Capacity", "smooth_capacity"], title = "Capacity degradation curve of the battery, its smoothed version and its end-of-life threshold")

    # Add horizontal line
    plot.add_hline(y = 1.4, line_width = 3, line_dash = "dash", line_color = "green")

    # Add legend for the horizontal line
    plot.add_trace(go.Scatter(x = [None], y = [None], mode = "lines", line = dict(color="green", width=3, dash="dash"), name = "End-of-life criteria"))

.. ipython:: python
    :suppress:

    import plotly.graph_objects as go

    plot = d_cap.plot(ts = "discharge_id", columns = ["Capacity", "smooth_capacity"], title = "Capacity degradation curve of the battery, its smoothed version and its end-of-life threshold")

    # Add horizontal line
    plot.add_hline(y = 1.4, line_width = 3, line_dash = "dash", line_color = "green")

    # Add legend for the horizontal line
    plot.add_trace(go.Scatter(x = [None], y = [None], mode = "lines", line = dict(color="green", width=3, dash="dash"), name = "End-of-life criteria"))
    fig = plot
    fig.write_html("SPHINX_DIRECTORY/figures/examples_battery_discharge_plotly_plote.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_discharge_plotly_plote.html

The sudden increases in battery capacity come from the self-charging property of Li-ion batteries. The smoothed graph makes 
the downward trend in the battery's capacity very clear.

An important observation here is that the battery meets the EOL criteria around the 125th cycle.

Goal and Problem Modeling
--------------------------

Understanding battery health is important, but at the time of writing, there's no direct way to measure it. In our case, we'll create a degredation model to find the relationship between a battery's overall health and the other properties in the dataset, which includes charge and discharge cycle duration, average voltage and current, etc.

One possible definition of the battery's overall health ("state of health" or "SoH") is the following:

Let :math:`Cap_{rate}` be the rated capacity of the battery when it's new (2Ah in our case), and :math:`Cap_{actual}` be the actual capacity of the battery at a specific time. The state of health of the battery is defined as:

.. math::

    SoH = \frac{Cap_{actual}}{Cap_{rate}} \times 100\% = \frac{1}{2}Cap_{actual}

Data preparation
-----------------

Outliet detection
++++++++++++++++++

Let's start by finding and removing the global outliers from our dataset.

.. code-block:: python

    battery5.outliers(
        columns = [
            "Voltage_measured",
            "Current_measured",
            "Temperature_measured","Capacity",
        ],
        name = "global_outlier",
        threshold = 4.0,
    )
    battery5.filter("global_outlier = 0").drop("global_outlier")

.. ipython:: python
    :suppress:

    battery5.outliers(
        columns = [
            "Voltage_measured",
            "Current_measured",
            "Temperature_measured",
            "Capacity",
        ],
        name = "global_outlier",
        threshold = 4.0,
    )
    battery5.filter("global_outlier = 0").drop("global_outlier")

Feature engineering
++++++++++++++++++++

Since measurements like voltage and temperature tend to differ within the different cycles, we'll create some features that can describe those cycles.

.. code-block:: python

    sample_cycle = battery5[battery5["Capacity"] == "1.83514614292266"]
    sample_cycle["Voltage_measured"].plot(ts = "Time")
    sample_cycle["Temperature_measured"].plot(ts = "Time")

.. ipython:: python
    :suppress:

    sample_cycle = battery5[battery5["Capacity"] == "1.83514614292266"]
    sample_cycle["Voltage_measured"].plot(ts = "Time")
    fig = sample_cycle["Temperature_measured"].plot(ts = "Time")
    fig.write_html("SPHINX_DIRECTORY/figures/examples_battery_temp_plot.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_temp_plot.html

We'll define new features that describe the minimum and maximum temperature during one cycle; the minimal voltage; and the time needed to reach minimum voltage and maximum temperature.

.. code-block:: python

    # filter for discharge cycles
    discharging_data = battery5[battery5["type"] == "discharge"]

    # define new features
    discharge_cycle_metrics = discharging_data.groupby(
            columns = ["start_time"], 
            expr = [
                "MIN(Temperature_measured) AS min_temp",
                "MAX(Temperature_measured) AS max_temp",
                "MIN(Voltage_measured) AS min_volt",
            ]
    ).join(
            discharging_data, 
            how = "left",
            on = {"min_volt": "voltage_measured"},
            expr1 = ["*"],
            expr2 = ["Time AS time_to_reach_minvolt"],
    ).join(
            discharging_data, 
            how = "left",
            on = {"max_temp": "temperature_measured"},
            expr1 = ["*"],
            expr2 = ["Time AS time_to_reach_maxtemp"],
    )

    # calculate values of SOH
    discharging_data = discharging_data.groupby(["start_time", "Capacity"])
    discharging_data["SOH"] = discharging_data["Capacity"] * 0.5

    # define the final dataset and save it to db
    final_df = discharge_cycle_metrics.join(
        discharging_data,
        on_interpolate = {"start_time": "start_time"},
        how = "left",
        expr1 = ["*"],
        expr2 = ["SOH AS SOH"],
    )

    # normalize the features
    final_df.normalize(
        method = "minmax",
        columns = [
            "min_temp",
            "max_temp",
            "min_volt",
            "time_to_reach_minvolt",
            "time_to_reach_maxtemp",
        ],
    )

    # save it to db
    final_df.to_db(name = "battery_data.finaldata_battery_5")

.. ipython:: python
    :suppress:

    # filter for discharge cycles
    discharging_data = battery5[battery5["type"] == "discharge"]

    # define new features
    discharge_cycle_metrics = discharging_data.groupby(
            columns = ["start_time"], 
            expr = [
                "MIN(Temperature_measured) AS min_temp",
                "MAX(Temperature_measured) AS max_temp",
                "MIN(Voltage_measured) AS min_volt",
            ]
    ).join(
            discharging_data, 
            how = "left",
            on = {"min_volt": "voltage_measured"},
            expr1 = ["*"],
            expr2 = ["Time AS time_to_reach_minvolt"],
    ).join(
            discharging_data, 
            how = "left",
            on = {"max_temp": "temperature_measured"},
            expr1 = ["*"],
            expr2 = ["Time AS time_to_reach_maxtemp"],
    )

    # calculate values of SOH
    discharging_data = discharging_data.groupby(["start_time", "Capacity"])
    discharging_data["SOH"] = discharging_data["Capacity"] * 0.5

    # define the final dataset and save it to db
    final_df = discharge_cycle_metrics.join(
        discharging_data,
        on_interpolate = {"start_time": "start_time"},
        how = "left",
        expr1 = ["*"],
        expr2 = ["SOH AS SOH"],
    )

    # normalize the features
    final_df.normalize(
        method = "minmax",
        columns = [
            "min_temp",
            "max_temp",
            "min_volt",
            "time_to_reach_minvolt",
            "time_to_reach_maxtemp",
        ],
    )

    # save it to db
    vp.drop("battery_data.finaldata_battery_5")
    final_df.to_db(name = "battery_data.finaldata_battery_5")

Machine Learning
-----------------

:py:mod:`~verticapy.machine_learning.vertica.automl.AutoML` tests several models and returns input scores for each. We can use this to find the best model for our dataset.

.. note:: We are only using the three algorithms, but you can change the ``estimator`` parameter to try all the ``native`` algorithms: ``estimator = 'native' ``.

.. code-block:: python

    from verticapy.machine_learning.vertica.automl import AutoML
    from verticapy.machine_learning.vertica import LinearRegression, RandomForestRegressor, Ridge

    model = AutoML(
        "battery_data.battery_autoML", 
        estimator = [
            RandomForestRegressor(),
            LinearRegression(),
            Ridge(),
        ],
        estimator_type = "regressor"
    )
    model.fit(
        "battery_data.finaldata_battery_5", 
        X = [
            "min_temp",
            "max_temp",
            "min_volt",
            "time_to_reach_minvolt",
            "time_to_reach_maxtemp",
        ],
        y = "SOH",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica.automl import AutoML
    from verticapy.machine_learning.vertica import LinearRegression, RandomForestRegressor, Ridge

    vp.drop("battery_data.battery_autoML")
    model = AutoML(
        "battery_data.battery_autoML", 
        estimator = [
            RandomForestRegressor(),
            LinearRegression(),
            Ridge(),
        ],
        estimator_type = "regressor"
    )
    model.fit(
        "battery_data.finaldata_battery_5", 
        X = [
            "min_temp",
            "max_temp",
            "min_volt",
            "time_to_reach_minvolt",
            "time_to_reach_maxtemp",
        ],
        y = "SOH",
    )

We can visualize the performance and efficency differences of each model with a plot.

.. code-block::

    model.plot()

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.plot()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_battery_auto_ml_plot.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_auto_ml_plot.html

.. ipython:: python

    # take the best model and its parameters
    best_model = model.best_model_
    params = best_model.get_params()
    print(best_model._model_type)

We can now define the model using those hyperparameters and train it.

.. code-block:: python

    # define a regression model based on the selected parameters
    model_rf = LinearRegression(name = "btr_lr1", **params)
    model_rf.fit(
        final_df,
        X = [
            "min_temp",
            "max_temp",
            "min_volt",
            "time_to_reach_minvolt",
            "time_to_reach_maxtemp",
        ],
        y = "SOH",
    )

.. ipython:: python
    :suppress:

    # define a regression model based on the selected parameters
    if "n_estimators" in params:
        params.pop("n_estimators")
    if "C" in params:
        params.pop("C")
    if "max_features" in params:
        params.pop("max_features")
    if "max_leaf_nodes" in params:
        params.pop("max_leaf_nodes")        
    vp.drop("btr_lr1")
    model_rf = LinearRegression(name = "btr_lr1", **params)
    model_rf.fit(
        final_df,
        X = [
            "min_temp",
            "max_temp",
            "min_volt",
            "time_to_reach_minvolt",
            "time_to_reach_maxtemp",
        ],
        y = "SOH",
    )

.. code-block:: python

    model_rf.regression_report()

.. ipython:: python
    :suppress:

    res = model_rf.regression_report()
    html_file = open("SPHINX_DIRECTORY/figures/examples_battery_reg_reprot.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_reg_reprot.html

The predictive power of our model looks pretty good. Let's use our model to predict the SoH of the battery. We can visualize our prediction with a plot against the true values.

.. code-block:: python

    # take the predicted values and the plot them along the true ones
    result = model_rf.predict(
        final_df, 
        name = "SOH_estimates",
    )
    result.plot(
        ts = "start_time", 
        columns = ["SOH", "SOH_estimates"],
    )

.. ipython:: python
    :suppress:
    :okwarning:

    result = model_rf.predict(
        final_df, 
        name = "SOH_estimates"
    )
    fig = result.plot(
        ts = "start_time", 
        columns = ["SOH", "SOH_estimates"],
    )
    fig.write_html("SPHINX_DIRECTORY/figures/examples_battery_auto_ml_plot.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_battery_auto_ml_plot.html

Conclusion
-----------

We successfully defined a battery degradation model that can make accurate predictions about the health of a Li-ion battery. This model could be used to, for example, accurately send warnings to users when their batteries meet the EOL criteria.