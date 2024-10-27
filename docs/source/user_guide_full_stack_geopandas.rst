.. _user_guide.full_stack.geopandas:

===========================
Integrating with GeoPandas
===========================

As of version 0.4.0, VerticaPy features GeoPandas integration. This allows you to easily export a :py:mod:`~verticapy.vDataFrame` as a GeoPandas DataFrame, giving you more control over geospatial data.

This example demonstrates the advantages of GeoPandas integration with the ``world`` dataset.

.. code-block:: python

    import verticapy as vp
    from verticapy.datasets import load_world

    # Setting the plotting lib
    vp.set_option("plotting_lib", "matplotlib")

    world = load_world()
    world.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    from verticapy.datasets import load_world
    # Setting the plotting lib
    vp.set_option("plotting_lib", "matplotlib")
    world = load_world()
    res = world.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_gpd_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_gpd_1.html

The :py:func:`~verticapy.vDataFrame.apply` function of the VerticaPy stats module allows you to apply any Vertica function to the data. Let's compute the area of each country.

.. code-block:: python

    import verticapy.sql.functions as fun

    world["geography"] = fun.apply("stv_geography", world["geometry"])
    world["geography"].astype("geography")
    world["area"] = fun.apply("st_area", world["geography"])
    world.head(100)

.. ipython:: python
    :suppress:

    import verticapy.sql.functions as fun

    world["geography"] = fun.apply("stv_geography", world["geometry"])
    world["geography"].astype("geography")
    world["area"] = fun.apply("st_area", world["geography"])
    res = world.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_gpd_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_gpd_2.html

We can now export our vDataFrame as a GeoPandas DataFrame.

.. ipython:: python

    df = world.to_geopandas(geometry = "geometry")
    df.head(200)

From there, we can draw any geospatial object.

.. ipython:: python
    :okwarning:

    ax = df.plot(
        edgecolor = "black",
        color = "white",
        figsize = (10, 9),
    )
    @savefig ug_fs_plot_gpd_3.png
    ax.set_title("World Map")

.. ipython:: python
    :okwarning:

    from verticapy.datasets import load_cities

    # Loading the cities dataset
    cities = load_cities()

    import matplotlib.pyplot as plt

    # Creating a Matplotlib figure
    fig, ax = plt.subplots()
    fig.set_size_inches(11, 8)

    # Extracting longitude and latitude
    cities["lon"] = fun.apply("st_x", cities["geometry"])
    cities["lat"] = fun.apply("st_y", cities["geometry"])

    # Drawing the data on a Map
    ax = cities.scatter(["lon", "lat"], ax = ax)

    @savefig ug_fs_plot_gpd_4.png
    df.plot(
        edgecolor = "black",
        color = "white",
        ax = ax,
    )

You can also draw maps using the :py:func:`~verticapy.vDataFrame.geo_plot` method.

.. ipython:: python
    :okwarning:

    from verticapy.datasets import load_africa_education

    # Africa Dataset
    africa = load_africa_education()
    africa_world = load_world()
    africa_world = africa_world[africa_world["continent"] == "Africa"]
    ax = africa_world["geometry"].geo_plot(
        color = "white",
        edgecolor = 'black',
    )

    # displaying schools in Africa
    @savefig ug_fs_plot_gpd_5.png
    africa.scatter(
        ["lon", "lat"],
        by = "country_long",
        ax = ax,
        max_cardinality = 20,
    )