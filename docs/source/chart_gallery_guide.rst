.. _chart_gallery.guide:

=========================
Chart Gallery User Guide
=========================

Introduction
------------

The Chart Gallery is an invaluable resource that allows you to delve into the art of creating diverse charts using a variety of supported libraries. These libraries include `Matplotlib`, `Highcharts`, and `Plotly`, each offering its unique advantages in terms of visualization and interactivity. However, it's not just about creating pretty pictures – it's about understanding what happens under the hood as these charts are generated.

Here, you'll gain insights into the inner workings of the chart generation process. You'll learn how Vertica is harnessed to perform complex calculations and aggregations that drive these charts. This understanding empowers you to craft charts that not only look great but also accurately represent your data.

In addition to demystifying the magic behind the scenes, we'll explore the art of parameter tuning. Each chart may have specific parameters that can be fine-tuned to meet your requirements. We'll guide you through these settings, helping you make informed decisions about how to tailor your charts for maximum impact.

Our Chart Gallery is filled with meticulously detailed examples, showcasing the vast array of charts that you can create with VerticaPy. Whether you're interested in creating insightful bar charts, interactive line plots, or visually stunning heatmaps, you'll find examples to inspire and guide your data visualization journey.

Please note that while we'll provide general principles and best practices in this guide, exploring the Chart Gallery is the best way to see these concepts in action. Dive in, experiment, and discover the limitless possibilities of data visualization with VerticaPy.

Switching Between Libraries
---------------------------

VerticaPy provides flexibility by allowing you to choose among different charting libraries: Matplotlib, Highcharts, and Plotly. Depending on your needs and preferences, you can switch between these libraries when creating charts.

Let's begin by importing `VerticaPy`.

.. ipython:: python

    import verticapy as vp

Please click on the tabs to explore how you can seamlessly switch between different libraries.

.. tab:: Plotly

    We can switch to using the `plotly` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "plotly")

.. tab:: Highcharts

    We can switch to using the `highcharts` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "highcharts")

.. tab:: Matplotlib

    We can switch to using the `matplotlib` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "matplotlib")

Data Sources and Chart Types
------------------------------

When it comes to creating charts with VerticaPy, you have two flexible options at your disposal:

1. **vDataFrame - The Python Object**: vDataFrame is a powerful Python object that simplifies the process of chart creation. It's been meticulously optimized to streamline your workflow. By utilizing vDataFrame, you'll benefit from the automatic generation of SQL queries that fetch the necessary data for your charts. This approach offers convenience and efficiency, as VerticaPy takes care of the complex SQL generation behind the scenes.

2. **SQL Queries**: Alternatively, you can opt to craft your own SQL queries directly within your Jupyter notebook magic cells. This gives you full control over the data retrieval process. Once you've executed the SQL query, VerticaPy will employ the returned results to generate your final chart. This approach provides ultimate flexibility, allowing you to fine-tune your queries to suit your specific charting requirements.

With these two distinct approaches, VerticaPy empowers you to seamlessly create charts that align with your data visualization needs. Whether you prefer the convenience of vDataFrame or the precision of handcrafted SQL queries, VerticaPy ensures that you can visualize your data effortlessly.

.. ipython:: python
    :suppress:

    import verticapy as vp
    import numpy as np

    N = 100 # Number of Records

    data = vp.vDataFrame({
      "score1": np.random.normal(5, 1, N),
      "score2": np.random.normal(8, 1.5, N),
      "score3": np.random.normal(10, 2, N),
      "category1": [np.random.choice(['A','B','C','D','E']) for _ in range(N)],
      "category2": [np.random.choice(['F','G','H']) for _ in range(N)],
    })

    vp.set_option("plotting_lib", "plotly")

Let's also import `numpy` to create a random dataset.

.. ipython:: python

    import numpy as np

Let's generate a dataset using the following data.

.. code-block:: python

	N = 100 # Number of Records

    data = vp.vDataFrame({
      "score1": np.random.normal(5, 1, N),
      "score2": np.random.normal(8, 1.5, N),
      "score3": np.random.normal(10, 2, N),
      "category1": [np.random.choice(['A','B','C']) for _ in range(N)],
      "category2": [np.random.choice(['D','E']) for _ in range(N)],
    })

In this dataset, we have two categorical columns and three numerical columns. We will use it in both Python and SQL statements.

Drawing a chart using vDataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Python, the process is straightforward. We can use various vDataFrame methods. For example, to draw a histogram of score1, you can simply call the `hist` method.

.. code-block:: python
          
    data["score1"].hist()

.. ipython:: python
	:suppress:
    :okwarning:

	fig = data["score1"].hist(width = 570)
	fig.write_html("figures/plotting_plotly_chart_gallery_hist_single.html")

.. raw:: html
	:file: SPHINX_DIRECTORY/figures/plotting_plotly_chart_gallery_hist_single.html

Drawing a chart using SQL Chart Magic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For SQL users, the `chart` magic extension allows you to create graphics.

We load the VerticaPy `chart` extension.

.. code-block:: python

    %load_ext verticapy.chart

In Python, the histogram interval `h` is automatically computed, while in SQL, you need to manually specify the binning for the chart. Additionally, in magic cells, you can use the operator `:` to indicate that you want to use a Python variable, and then assign a value to `h`.

.. code-block:: python

	h = 2

We write the SQL query using Jupyter magic cells. You can change the type of plots using the `k` option.

.. code-block:: sql
    
    %%chart -k hist
    SELECT 
        FLOOR(score1 / :h) * :h AS score1, 
        COUNT(*) / :N AS density
    FROM :data 
    GROUP BY 1
    ORDER BY 1;

.. ipython:: python
	:suppress:
    :okwarning:

	fig = data["score1"].hist(h = 2, width = 570)
	fig.write_html("figures/plotting_plotly_chart_gallery_hist_single_h10.html")

.. raw:: html
	:file: SPHINX_DIRECTORY/figures/plotting_plotly_chart_gallery_hist_single_h10.html

Understanding Parameters
-------------------------

The Chart Gallery offers various parameters to customize your visualizations. Two important parameters to understand are:

1. `h` (Bar Bin Interval): In 1D and 2D graphics, `h` represents the bar bin interval. VerticaPy automatically computes this value, but you can also choose it based on your data characteristics.

2. `max_cardinality`: This parameter controls the maximum number of categories to display in charts. Understanding how to set this parameter is crucial for creating informative visualizations.

Bar Bin Interval: `h`
~~~~~~~~~~~~~~~~~~~~~

`h` is a crucial parameter as it determines how numerical columns are binned. In our example, we can bin 'score1' and 'score2'. If no values are entered, VerticaPy will use methods like Sturges and Freedman Diaconis to determine the bin size for these two numerical features. Alternatively, you can specify a tuple (h1, h2) to set custom bin sizes.

.. code-block:: python
          
    data.bar(columns = ["score1", "score2"], h = (2, 3))

.. ipython:: python
	:suppress:
    :okwarning:

	fig = data.bar(columns = ["score1", "score2"], h = (2, 3))
	fig.write_html("figures/plotting_plotly_chart_gallery_bar_h1_h2.html")

.. raw:: html
	:file: SPHINX_DIRECTORY/figures/plotting_plotly_chart_gallery_bar_h1_h2.html

Max Cardinality
~~~~~~~~~~~~~~~

`max_cardinality` is a parameter that allows you to display only important categories. It represents the maximum number of distinct elements for a column to be considered categorical. Less frequent elements are grouped together into a new category called 'Others'.

For example, if 'category1' has 5 distinct elements, you can use `max_cardinality` to filter and keep only two of those categories.

.. code-block:: python
          
    data.scatter(columns = ["score1", "score2"], by = "category1", max_cardinality = 2)

.. ipython:: python
	:suppress:
    :okwarning:

	fig = data.scatter(columns = ["score1", "score2"], by = "category1", max_cardinality = 2)
	fig.write_html("figures/plotting_plotly_chart_gallery_scatter_max_cardinality.html")

.. raw:: html
	:file: SPHINX_DIRECTORY/figures/plotting_plotly_chart_gallery_scatter_max_cardinality.html

You can also utilize the `cat_priority` parameter to filter and display only the specific categories you need.

.. code-block:: python
          
    data.scatter(columns = ["score1", "score2"], by = "category1", cat_priority = ["C", "D"])

.. ipython:: python
	:suppress:
    :okwarning:

	fig = data.scatter(columns = ["score1", "score2"], by = "category1", cat_priority = ["C", "D"])
	fig.write_html("figures/plotting_plotly_chart_gallery_scatter_max_cardinality.html")

.. raw:: html
	:file: SPHINX_DIRECTORY/figures/plotting_plotly_chart_gallery_scatter_max_cardinality.html


Data Filtering and Processing
-----------------------------

In the world of chart creation with VerticaPy, it's essential to understand that not all graphics are created equal. Some charts can be efficiently computed thanks to data aggregation, while others require the entire dataset. For instance, bar graphs, pie charts, histograms, and more can be seamlessly computed using aggregation. VerticaPy leverages the power of Vertica to push SQL statements, gather the necessary data, and generate these charts with ease. 

However, when it comes to scatter plots and line graphs, things get a bit more intricate. These chart types demand the entire dataset, which may be substantial in size. In such cases, VerticaPy employs downsampling techniques to ensure efficient processing. The tool provides numerous parameters for fine-tuning the downsampling process, allowing you to strike the perfect balance between data representation and performance.

Filtering Data for Scatter Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many charts utilize scatter plots, which may include a `max_nb_points` parameter. You can employ this parameter to control the number of data points displayed. In such cases, VerticaPy employs a hybrid downsampling approach.

.. code-block:: python
          
    data.scatter(columns = ["score1", "score2"], by = "category1", max_nb_points = 30)

.. ipython:: python
	:suppress:
    :okwarning:

	fig = data.scatter(columns = ["score1", "score2"], by = "category1", max_cardinality = 2)
	fig.write_html("figures/plotting_plotly_chart_gallery_scatter_max_cardinality.html")

.. raw:: html
	:file: SPHINX_DIRECTORY/figures/plotting_plotly_chart_gallery_scatter_max_cardinality.html

Filtering Data for Time Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For time series plots, you can filter the data using two numerical or timestamp parameters: `start_date` and `end_date`.

Let's use the following dataset.

.. ipython:: python

    N = 30 # Number of records

    data = vp.vDataFrame({
        "date": [1990 + i for i in range(N)],
        "population": [200 + i ** 2 - 3 * i for i in range(N)],
    })

Let's create a chart by filtering the data using two dates.

.. code-block:: python
          
    data["population"].plot(ts = "date", start_date = 1995, end_date = 2010)

.. ipython:: python
	:suppress:
    :okwarning:

	fig = data["population"].plot(ts = "date", start_date = 1995, end_date = 2010)
	fig.write_html("figures/plotting_plotly_chart_gallery_line_filter.html")

.. raw:: html
	:file: SPHINX_DIRECTORY/figures/plotting_plotly_chart_gallery_line_filter.html

Chart Customization
-------------------

Complete examples are available on the various chart pages.

.. hint::

    For SQL users who use Jupyter Magic cells, chart customization must be done in Python. They can then export the graphic using the last magic cell result.

    .. code-block:: python

        chart = _

    Now, the chart variable includes the graphic. Depending on the library you are using, you will obtain a different object.

Each chart function returns a graphic that can be customized using the source library.

.. Important:: Different customization parameters are available for Plotly, Highcharts, and Matplotlib. 
    For a comprehensive list of customization features, please consult the documentation of the respective 
    libraries: `plotly <https://plotly.com/python-api-reference/>`_, `matplotlib <https://matplotlib.org/stable/api/matplotlib_configuration_api.html>`_ and `highcharts <https://api.highcharts.com/highcharts/>`_.

Conclusion
-----------

The Chart Gallery in VerticaPy is a versatile tool for creating interactive and informative visualizations. With this guide, you can navigate the different options, customize your charts, and make data-driven decisions.

Learn More
~~~~~~~~~~

For in-depth tutorials, code samples, and documentation, visit the `Chart Gallery Home Page` :ref:`chart_gallery`.

We hope this guide helps you harness the full potential of the Chart Gallery for your data visualization needs.