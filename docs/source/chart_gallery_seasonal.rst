.. _chart_gallery.seasonal:

=======================
Seasonal Decompose Plot
=======================

.. Necessary Code Elements

.. ipython:: python
    :suppress:

    import verticapy as vp
    import verticapy.datasets as vpd
    import verticapy.machine_learning.model_selection.statistical_tests as vmlt

    passengers = vpd.load_airline_passengers()

    # We use seasonal decompose to get the time series component
    decomposition = vmlt.seasonal_decompose(
        input_relation = passengers, 
        columns = "passengers", 
        ts = "date",
        polynomial_order = 2,
        mult = True,
        use_row = False,
    )


General
-------

Let's begin by importing the dataset module of `VerticaPy`. It provides a range of datasets for both training and exploring VerticaPy's capabilities.

.. code-block:: python

    import verticapy.datasets as vpd


Let's utilize the Airline Passenger dataset to demonstrate time series capabilities.

.. code-block:: python
    
    import verticapy.datasets as vpd

    passengers = vpd.load_airline_passengers()

This dataset is well-suited for seasonal decomposition. It represents the time series of the number of passengers for a specific flight since 1950. Notably, it exhibits a noticeable trend and seasonality pattern. It is evident that the time series follows a multiplicative model.

Let's perform a decomposition of the time series.

But before that let's impor the VerticaPy ML tests.

.. code-block:: python

    import verticapy.machine_learning.model_selection.statistical_tests as vmlt


.. code-block:: python
    
    # We use seasonal decompose to get the time series component
    decomposition = vms.seasonal_decompose(
        input_relation = passengers, 
        columns = "passengers", 
        ts = "date",
        polynomial_order = 2,
        mult = True,
        use_row = False,
    )

To create a seasonal decomposition plot, we must visualize the primary time series along with all its individual components. This decomposition process extracts various time series components, and we can then proceed to visualize each of them separately.

.. note::
    
    Subplots are not available in Highcharts. Therefore, we will demonstrate how to create a seasonal decomposition plot using Plotly and Matplotlib.

.. ipython:: python
    :suppress:

    import verticapy as vp

.. tab:: Plotly

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")

    We can switch to using the `plotly` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "plotly")

    We can first create custom subplot array using `plotly`. The dimensions are set to 3 x 2.

    .. code-block:: python

        from plotly.subplots import make_subplots

        fig = make_subplots(rows=3, cols=2, column_widths=[0.7, 0.3], 
                            specs=[[{"rowspan": 3}, {"rowspan": 1}],
                                    [{}, {"rowspan": 1}],
                                    [{}, {"rowspan": 1}]],)


    Then we can indivually add the traces.

    .. code-block:: python
        
        # Add the first trace (spans three rows)
        fig.add_trace(
            decomposition["passengers"].plot(ts = "date", colors = "#0073E7").data[0],
            row=1, col=1,
        )

        # Add the second trace (second column, first row)
        fig.add_trace(
            decomposition["passengers_trend"].plot(ts = "date", colors = "black").data[0],
            row=1, col=2
        )

        # Add the third trace (second column, second row)
        fig.add_trace(
            decomposition["passengers_seasonal"].plot(ts = "date", colors = "green").data[0],
            row=2, col=2
        )

        # Add the fourth trace (third row, second column)
        fig.add_trace(
            decomposition["passengers_epsilon"].plot(ts = "date",colors = "grey").data[0],
            row=3, col=2
        )
        fig.update_layout(height = 500, width = 700)


    .. ipython:: python
        :suppress:

        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=2, column_widths=[0.7, 0.3], 
                            specs=[[{"rowspan": 3}, {"rowspan": 1}],
                                    [{}, {"rowspan": 1}],
                                    [{}, {"rowspan": 1}]],)

        fig.add_trace(
            decomposition["passengers"].plot(ts = "date", colors = "#0073E7").data[0],
            row=1, col=1,
        )

        # Add the second trace (second column, first row)
        fig.add_trace(
            decomposition["passengers_trend"].plot(ts = "date", colors = "black").data[0],
            row=1, col=2
        )

        # Add the third trace (second column, second row)
        fig.add_trace(
            decomposition["passengers_seasonal"].plot(ts = "date", colors = "green").data[0],
            row=2, col=2
        )

        # Add the fourth trace (third row, second column)
        fig.add_trace(
            decomposition["passengers_epsilon"].plot(ts = "date",colors = "grey").data[0],
            row=3, col=2
        )
        fig.update_layout(
            height = 500, 
            width = 700,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.write_html("figures/plotting_plotly_seasonal.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_seasonal.html


            
.. tab:: Matplotlib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

        import matplotlib.pyplot as plt

        fig = plt.figure()
        fig.set_size_inches(10, 6)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(324)
        ax4 = fig.add_subplot(326)

    We can switch to using the `matplotlib` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "matplotlib")

    We need to import `matplotlib.pyplot`.

    .. code-block:: python

        import matplotlib.pyplot as plt

    We'll create four subplots for this purpose.

    .. code-block:: python

        fig = plt.figure()
        fig.set_size_inches(10, 6)

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(324)
        ax4 = fig.add_subplot(326)

    Following this, we can proceed to plot the final visualization.

    .. ipython:: python
        :okwarning:

        decomposition["passengers"].plot(ts = "date", ax = ax1, color = "#0073E7")
        decomposition["passengers_trend"].plot(ts = "date", ax = ax2, color = "black")
        ax2.set_xlabel("")
        ax2.get_xaxis().set_ticks([])
        decomposition["passengers_seasonal"].plot(ts = "date", ax = ax3, color = "green")
        ax3.set_xlabel("")
        ax3.get_xaxis().set_ticks([])
        decomposition["passengers_epsilon"].plot(ts = "date", ax = ax4, color = "grey")
        plt.savefig("figures/plotting_matplotlib_seasonal.png")


    .. image:: ../../../docs/figures/plotting_matplotlib_seasonal.png
        :width: 100%
        :align: center


___________________


Chart Customization
-------------------

VerticaPy empowers users with a high degree of flexibility when it comes to tailoring the visual aspects of their plots. 
This customization extends to essential elements such as **color schemes**, **text labels**, and **plot sizes**, as well as a wide range of other attributes that can be fine-tuned to align with specific design preferences and analytical requirements. Whether you want to make your visualizations more visually appealing or need to convey specific insights with precision, VerticaPy's customization options enable you to craft graphics that suit your exact needs.

.. note:: As seasonal decomposition plots consist of multiple line charts, we recommend referring to the page on customizing :ref:`line` charts for guidance on customization.
