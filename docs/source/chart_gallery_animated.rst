.. _chart_gallery.animated:

==============
Animated Plots
==============

.. Necessary Code Elements

.. ipython:: python
    :suppress:

    import verticapy as vp
    import verticapy.datasets as vpd

    # Population Growth Dataset - Ideal for BAR Race | Animated PIE
    pop_growth = vpd.load_pop_growth()
    # Gap Minder Dataset - Ideal for Animated Bubble
    gapminder = vpd.load_gapminder()
    # Commodities Dataset - Ideal for Animated Time Series
    commodities = vpd.load_commodities()


General
-------

Let's begin by importing the dataset module of `VerticaPy`. It provides a range of datasets for both training and exploring VerticaPy's capabilities.

.. ipython:: python

    import verticapy.datasets as vpd

Let's leverage the various datasets to generate different types of animated plots.

.. code-block:: python
    
    import verticapy.datasets as vpd

    # Population Growth Dataset - Ideal for BAR Race | Animated PIE
    pop_growth = vpd.load_pop_growth()
    # Gap Minder Dataset - Ideal for Animated Bubble
    gapminder = vpd.load_gapminder()
    # Commodities Dataset - Ideal for Animated Time Series
    commodities = vpd.load_commodities()

VerticaPy's animated charts, including bar races, bubble animations, pie chart transitions, and time series animations, add a dynamic dimension to data visualization. These animated visualizations allow for the dynamic presentation of data trends, changes, and relationships, making complex information easily comprehensible and engaging. Whether tracking evolving data over time, showcasing multi-dimensional insights, or highlighting shifts in proportions, VerticaPy's animated charts transform data into captivating narratives for enhanced data exploration and communication.

.. hint::
    
    We will utilize various datasets within VerticaPy, which are abundant and accessible through the dataset module. These datasets cater to a wide range of use cases, providing versatile options for data analysis and experimentation.

.. ipython:: python
    :suppress:

    import verticapy as vp
            
.. tab:: Matplotlib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    We can switch to using the `matplotlib` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "matplotlib")

    VerticaPy's animated plots add dynamic storytelling to data visualization.

    .. tab:: Bar

      .. code-block:: python

          pop_growth.animated_bar(
            ts = "year",
            columns = ["city", "population"],
            by = "continent",
            start_date = 1970,
            end_date = 1980,
          )

      .. ipython:: python
          :suppress:

          fig = pop_growth.animated_bar(
            ts = "year",
            columns = ["city", "population"],
            by = "continent",
            start_date = 1970,
            end_date = 1980,
          )

          with open("figures/plotting_matplotlib_animated_bar.html", "w") as file:
            file.write(fig.__html__())

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_matplotlib_animated_bar.html

    .. tab:: Pie

      .. code-block:: python

          fig = pop_growth.animated_pie(
            ts = "year",
            columns = ["city", "population"],
            by = "continent",
            start_date = 1970,
            end_date = 1980,
          )

      .. ipython:: python
          :suppress:

          fig = pop_growth.animated_pie(
            ts = "year",
            columns = ["city", "population"],
            by = "continent",
            start_date = 1970,
            end_date = 1980,
          )

          with open("figures/plotting_matplotlib_animated_pie.html", "w") as file:
            file.write(fig.__html__())

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_matplotlib_animated_pie.html

    .. tab:: Bubble

      .. code-block:: python

          fig = gapminder.animated_scatter(
            ts = "year",
            columns = ["lifeExp", "gdpPercap", "country", "pop"],
            by = "continent",
            limit_labels=10, 
            limit_over=100
          )

      .. ipython:: python
          :suppress:

          fig = gapminder.animated_scatter(
            ts = "year",
            columns = ["lifeExp", "gdpPercap", "country", "pop"],
            by = "continent",
            limit_labels=10, 
            limit_over=100
          )

          with open("figures/plotting_matplotlib_animated_bubble.html", "w") as file:
            file.write(fig.__html__())

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_matplotlib_animated_bubble.html

    .. tab:: Time Series

      .. code-block:: python

          fig = commodities.animated_plot(ts = "date")

      .. ipython:: python
          :suppress:

          fig = commodities.animated_plot(ts = "date")

          with open("figures/plotting_matplotlib_animated_time.html", "w") as file:
            file.write(fig.__html__())

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_matplotlib_animated_time.html

___________________


Chart Customization
-------------------

VerticaPy empowers users with a high degree of flexibility when it comes to tailoring the visual aspects of their plots. 
This customization extends to essential elements such as **color schemes**, **text labels**, and **plot sizes**, as well as a wide range of other attributes that can be fine-tuned to align with specific design preferences and analytical requirements. Whether you want to make your visualizations more visually appealing or need to convey specific insights with precision, VerticaPy's customization options enable you to craft graphics that suit your exact needs.

.. note:: As animated plots encompass various chart types, including line, pie, and scatter plots, customization options may vary between these graphics. For detailed guidance on tailoring your visualization, please consult the corresponding section in the :ref:`chart_gallery`.
