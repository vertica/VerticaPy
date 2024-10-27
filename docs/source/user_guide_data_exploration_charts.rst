.. _user_guide.data_exploration.charts:

=======
Charts
=======

Charts are a powerful tool for understanding and interpreting data.

Most charts use aggregations to represent the dataset, and others downsample the data to represent a subset.

.. note:: See :ref:`chart_gallery` for all the different charts and their syntax.

First, let's import the modules needed for this notebook.

.. ipython:: python

  # VerticaPy
  import verticapy as vp
  from verticapy.datasets import load_titanic, load_iris, load_world, load_amazon, load_africa_education

  # Numpy & Matplotlib
  import numpy as np
  import matplotlib.pyplot as plt

Let's start with pies and histograms. Drawing the pie or histogram of a categorical column in VerticaPy is quite easy.

.. note:: You can conveniently switch between the three available plotting libraries using :py:func:`~verticapy.set_option`.

.. code-block::

  # Setting the plotting lib
  vp.set_option("plotting_lib", "highcharts")
  
  titanic = load_titanic()
  titanic["pclass"].bar()

.. ipython:: python
  :suppress:

  # Setting the plotting lib
  vp.set_option("plotting_lib", "highcharts")
  titanic = load_titanic()
  fig = titanic["pclass"].bar()
  html_text = fig.htmlcontent.replace("container", "user_guides_data_exploration_titanic_bar")
  with open("figures/user_guides_data_exploration_titanic_bar.html", "w") as file:
    file.write(html_text)

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_bar.html

.. code-block::

  titanic["pclass"].pie()

.. ipython:: python
  :suppress:

  fig = titanic["pclass"].pie()
  html_text = fig.htmlcontent.replace("container", "user_guides_data_exploration_titanic_pie")
  with open("figures/user_guides_data_exploration_titanic_pie.html", "w") as file:
    file.write(html_text)

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_pie.html

.. code-block::

  titanic["home.dest"].bar()

.. ipython:: python
  :suppress:

  fig = titanic["home.dest"].bar()
  html_text = fig.htmlcontent.replace("container", "user_guides_data_exploration_titanic_home_dest_bar")
  with open("figures/user_guides_data_exploration_titanic_home_dest_bar.html", "w") as file:
    file.write(html_text)

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_home_dest_bar.html

These methods will draw the most occurent categories and merge the others. To change the number of elements, you can use the ``max_cardinality`` parameter.

.. code-block::

  titanic["home.dest"].bar(max_cardinality = 5)

.. ipython:: python
  :suppress:

  fig = titanic["home.dest"].bar(max_cardinality = 5)
  html_text = fig.htmlcontent.replace("container", "user_guides_data_exploration_titanic_home_dest_bar_max_cardinality")
  with open("figures/user_guides_data_exploration_titanic_home_dest_bar_max_cardinality.html", "w") as file:
    file.write(html_text)

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_home_dest_bar_max_cardinality.html

When dealing with numerical data types, the process is different. Vertica needs to discretize the numerical features to draw them. You can choose the bar width (``h`` parameter) or let VerticaPy compute an optimal width using the Freedman-Diaconis rule.

.. code-block::

  titanic["age"].hist()

.. ipython:: python
  :suppress:

  fig = titanic["age"].hist()
  html_text = fig.htmlcontent.replace("container", "user_guides_data_exploration_titanic_age_hist")
  with open("figures/user_guides_data_exploration_titanic_age_hist.html", "w") as file:
    file.write(html_text)

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_age_hist.html

.. code-block::

  titanic["age"].hist(h = 5)

.. ipython:: python
  :suppress:

  fig = titanic["age"].hist(h = 5)
  html_text = fig.htmlcontent.replace("container", "user_guides_data_exploration_titanic_age_hist_h5")
  with open("figures/user_guides_data_exploration_titanic_age_hist_h5.html", "w") as file:
    file.write(html_text)

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_age_hist_h5.html

You can also change the occurences by another aggregation with the ``method`` and ``of`` parameters.

.. code-block::

  titanic["age"].hist(method = "avg", of = "survived")

.. ipython:: python
  :suppress:

  fig = titanic["age"].hist(method = "avg", of = "survived")
  html_text = fig.htmlcontent.replace("container", "user_guides_data_exploration_titanic_age_hist_avs")
  with open("figures/user_guides_data_exploration_titanic_age_hist_avs.html", "w") as file:
    file.write(html_text)

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_age_hist_avs.html

VerticaPy uses the same process for other graphics, like 2-dimensional histograms and bar charts.

Let us showcase another plotting library for these plots.

.. code-block::
  
  # Setting the plotting lib
  vp.set_option("plotting_lib", "plotly")

  titanic.bar(["pclass", "survived"])

.. ipython:: python
  :suppress:

  # Setting the plotting lib
  vp.set_option("plotting_lib", "plotly")
  fig = titanic.bar(["pclass", "survived"])
  fig.write_html("SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_bar_pclass_surv.html")

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_bar_pclass_surv.html

.. note:: VerticaPy has three main plotting libraries. Look at :ref:`chart_gallery` section for all the different plots.

.. code-block::
    
  titanic.hist(
      ["fare", "pclass"],
      method = "avg",
      of = "survived",
  )

.. ipython:: python
  :suppress:

  fig = titanic.hist(
      ["fare", "pclass"],
      method = "avg",
      of = "survived",
  )
  fig.write_html("SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_bar_pclass_fare.html")

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_bar_pclass_fare.html

Pivot tables give us aggregated information for every category and are more powerful than histograms or bar charts.

.. code-block::
    
  titanic.pivot_table(
      ["pclass", "fare"], 
      method = "avg",
      of = "survived",
      fill_none = np.nan,
  )

.. ipython:: python
  :suppress:
  :okwarning:

  fig = titanic.pivot_table(
      ["pclass", "fare"], 
      method = "avg",
      of = "survived",
      fill_none = np.nan,
  )
  fig.write_html("SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_bar_pclass_fare_fill.html")

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_bar_pclass_fare_fill.html

Box plots are useful for understanding statistical dispersion.

.. code-block::
    
  titanic.boxplot(columns = ["age", "fare"])

.. ipython:: python
  :suppress:
  :okwarning:

  fig = titanic.boxplot(columns = ["age", "fare"])
  fig.write_html("SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_boxplot.html")

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_boxplot.html

.. code-block::
    
  titanic["age"].boxplot()

.. ipython:: python
  :suppress:
  :okwarning:

  fig = titanic["age"].boxplot()
  fig.write_html("SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_boxplot_one.html")

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_titanic_boxplot_one.html

Scatter and bubble plots are also useful for identifying patterns in your data. Note, however, that these methods don't use aggregations; VerticaPy downsamples the data before plotting. You can use the ``max_nb_points`` to limit the number of points and avoid unnecessary memory usage.

.. code-block::
    
  iris = load_iris()
  iris.scatter(
      ["SepalLengthCm", "PetalWidthCm"], 
      by = "Species", 
      max_nb_points = 1000,
  )

.. ipython:: python
  :suppress:
  :okwarning:

  iris = load_iris()
  fig = iris.scatter(
      ["SepalLengthCm", "PetalWidthCm"], 
      by = "Species", 
      max_nb_points = 1000,
  )
  fig.write_html("SPHINX_DIRECTORY/figures/user_guides_data_exploration_iris_scatter.html")

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_iris_scatter.html

Now, let us look at a 3D scatter plot.

.. code-block::
    
  iris.scatter(
      ["SepalLengthCm", "PetalWidthCm", "SepalWidthCm"],
      by = "Species",
      max_nb_points = 1000,
  )

.. ipython:: python
  :suppress:
  :okwarning:

  fig = iris.scatter(
      ["SepalLengthCm", "PetalWidthCm", "SepalWidthCm"], 
      by = "Species", 
      max_nb_points = 1000,
  )
  fig.write_html("SPHINX_DIRECTORY/figures/user_guides_data_exploration_iris_scatter_3d.html")

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_iris_scatter_3d.html

Similarly, we can plot a bubble plot:

.. code-block::
    
  iris.scatter(
      ["SepalLengthCm", "PetalWidthCm"], 
      size = "SepalWidthCm",
      by = "Species",
      max_nb_points = 1000,
  )

.. ipython:: python
  :suppress:
  :okwarning:

  fig = iris.scatter(
      ["SepalLengthCm", "PetalWidthCm"], 
      size = "SepalWidthCm",
      by = "Species",
      max_nb_points = 1000,
  )
  fig.write_html("SPHINX_DIRECTORY/figures/user_guides_data_exploration_iris_scatter_bubble.html")

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_iris_scatter_bubble.html

For more information on scatter look at :py:func:`verticapy.vDataFrame.scatter`.

Hexbin plots can be useful for generating heatmaps. These summarize data in a similar way to scatter plots, but compute aggregations to get the final results.

.. ipython:: python
  
  # Setting the plotting lib
  vp.set_option("plotting_lib", "matplotlib")

  @savefig user_guides_data_exploration_iris_hexbin.png
  iris.hexbin(
      ["SepalLengthCm", "SepalWidthCm"], 
      method = "avg", 
      of = "PetalWidthCm",
  )

Hexbin, scatter, and bubble plots also allow you to provide a background image. The dataset used below is available here.

.. code-block:: python

  africa = load_africa_education()

  # displaying avg students score in Africa
  africa.hexbin(
      ["lon", "lat"],
      method = "avg",
      of = "zralocp",
      img = "img/africa.png",
  )

.. ipython:: python
  :suppress:

  africa = load_africa_education()

  # displaying avg students score in Africa
  @savefig user_guides_data_exploration_africa_hexbin.png
  africa.hexbin(
      ["lon", "lat"],
      method = "avg",
      of = "zralocp",
      img = "SPHINX_DIRECTORY/source/_static/website/user_guides/data_exploration/africa.png"
  )

It is also possible to use SHP datasets to draw maps.

.. ipython:: python

  # Africa Dataset
  africa_world = load_world()
  africa_world = africa_world[africa_world["continent"] == "Africa"]
  ax = africa_world["geometry"].geo_plot(
      color = "white",
      edgecolor = "black",
  );
  # displaying schools in Africa
  @savefig user_guides_data_exploration_africa_scatter.png
  africa.scatter(
      ["lon", "lat"],
      by = "country_long",
      ax = ax,
      max_cardinality = 100
  )

Time-series plots are also available with the :py:func:`~verticapy.vDataFrame.plot` method.

.. ipython:: python

  amazon = load_amazon();
  amazon.filter(amazon["state"]._in(["ACRE", "RIO DE JANEIRO", "PARÁ"]));
  @savefig user_guides_data_exploration_amazon_time.png
  amazon["number"].plot(ts = "date", by = "state")

Since time-series plots do not aggregate the data, it's important to choose the correct ``start_date`` and ``end_date``.

.. code-block:: python

  amazon["number"].plot(
      ts = "date", 
      by = "state", 
      start_date = "2010-01-01",
  )

.. ipython:: python
  :suppress:
  :okwarning:

  # Setting the plotting lib
  vp.set_option("plotting_lib", "plotly")

  fig = amazon["number"].plot(
      ts = "date", 
      by = "state", 
      start_date = "2010-01-01",
  )
  fig.write_html("SPHINX_DIRECTORY/figures/user_guides_data_exploration_amazon_time_plot.html")

.. raw:: html
  :file: SPHINX_DIRECTORY/figures/user_guides_data_exploration_amazon_time_plot.html