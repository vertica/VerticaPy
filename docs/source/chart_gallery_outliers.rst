.. _chart_gallery.outliers:

===========================
Machine Learning - Outliers
===========================

.. Necessary Code Elements

.. ipython:: python
    :suppress:

    import verticapy as vp
    import numpy as np

    N = 100 # Number of Records
    k = 10 # step

    # Normal Distributions
    x = np.random.normal(5, 1, round(N / 2))
    y = np.random.normal(3, 1, round(N / 2))

    # Creating a vDataFrame with two clusters
    data = vp.vDataFrame({
        "x": np.concatenate([x, x + k]),
        "y": np.concatenate([y, y + k]),
    })

General
-------

VerticaPy's outlier plots offer an essential means of identifying and comprehending outliers within your dataset. These plots provide valuable visual insights into data points that significantly deviate from the expected distribution, facilitating the detection of anomalies or potential data errors. Whether through box plots, scatter plots, or other visualizations, VerticaPy equips data analysts with powerful tools to enhance outlier detection and data quality assessment.

Let's begin by importing ``verticapy``.

.. ipython:: python

    import verticapy as vp

Let's also import ``numpy`` to create a random dataset.

.. ipython:: python

    import numpy as np

Let's generate a dataset using the following data.

.. code-block:: python
        
    N = 100 # Number of Records
    k = 10 # step

    # Normal Distributions
    x = np.random.normal(5, 1, round(N / 2))
    y = np.random.normal(3, 1, round(N / 2))

    # Creating a vDataFrame with two clusters
    data = vp.vDataFrame({
        "x": np.concatenate([x, x + k]),
        "y": np.concatenate([y, y + k]),
    })

In the context of data visualization, we have the flexibility to harness multiple plotting libraries to craft a wide range of graphical representations. VerticaPy, as a versatile tool, provides support for several graphic libraries, such as Matplotlib, Highcharts, and Plotly. Each of these libraries offers unique features and capabilities, allowing us to choose the most suitable one for our specific data visualization needs.

.. image:: ../../docs/source/_static/plotting_libs.png
   :width: 80%
   :align: center

.. note::
    
    To select the desired plotting library, we simply need to use the :py:func:`~verticapy.set_option` function. VerticaPy offers the flexibility to smoothly transition between different plotting libraries. In instances where a particular graphic is not supported by the chosen library or is not supported within the VerticaPy framework, the tool will automatically generate a warning and then switch to an alternative library where the graphic can be created.

Please click on the tabs to view the various graphics generated by the different plotting libraries.

.. ipython:: python
    :suppress:

    import verticapy as vp

.. tab:: Plotly

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")

    We can switch to using the ``plotly`` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "plotly")
    
    .. tab:: 1D

      .. code-block:: python
          
          data.outliers_plot(columns = ["x"])

      .. ipython:: python
          :suppress:
          :okwarning:
        
          fig = data.outliers_plot(columns = ["x"])
          fig.write_html("figures/plotting_plotly_outliers_1d_1.html")

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_plotly_outliers_1d_1.html

    .. tab:: 2D

      .. code-block:: python
          
          data.outliers_plot(columns = ["x", "y"])

      .. ipython:: python
          :suppress:
          :okwarning:
        
          fig = data.outliers_plot(columns = ["x", "y"])
          fig.write_html("figures/plotting_plotly_outliers_2d_1.html")

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_plotly_outliers_2d_1.html

.. tab:: Highcharts

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")

    We can switch to using the ``highcharts`` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "highcharts")

    .. tab:: 1D

      .. code-block:: python
          
          data.outliers_plot(columns = ["x"])

      .. ipython:: python
          :suppress:

          fig = data.outliers_plot(columns = ["x"])
          html_text = fig.htmlcontent.replace("container", "plotting_highcharts_outliers_1d_1")
          with open("figures/plotting_highcharts_outliers_1d_1.html", "w") as file:
            file.write(html_text)

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_highcharts_outliers_1d_1.html

    .. tab:: 2D

      .. code-block:: python
          
          data.outliers_plot(columns = ["x", "y"])

      .. ipython:: python
          :suppress:

          fig = data.outliers_plot(columns = ["x", "y"])
          html_text = fig.htmlcontent.replace("container", "plotting_highcharts_outliers_2d_1")
          with open("figures/plotting_highcharts_outliers_2d_1.html", "w") as file:
            file.write(html_text)

      .. raw:: html
          :file: SPHINX_DIRECTORY/figures/plotting_highcharts_outliers_2d_1.html

        
.. tab:: Matplotlib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    We can switch to using the ``matplotlib`` module.

    .. code-block:: python
        
        vp.set_option("plotting_lib", "matplotlib")

    .. tab:: 1D

      .. ipython:: python
          :okwarning:

          @savefig plotting_matplotlib_outliers_1d_1.png
          data.outliers_plot(columns = ["x"])

    .. tab:: 2D

      .. ipython:: python
          :okwarning:

          @savefig plotting_matplotlib_outliers_2d_1.png
          data.outliers_plot(columns = ["x", "y"])

___________________


Chart Customization
-------------------

VerticaPy empowers users with a high degree of flexibility when it comes to tailoring the visual aspects of their plots. 
This customization extends to essential elements such as **color schemes**, **text labels**, and **plot sizes**, as well as a wide range of other attributes that can be fine-tuned to align with specific design preferences and analytical requirements. Whether you want to make your visualizations more visually appealing or need to convey specific insights with precision, VerticaPy's customization options enable you to craft graphics that suit your exact needs.

.. Important:: Different customization parameters are available for Plotly, Highcharts, and Matplotlib. 
    For a comprehensive list of customization features, please consult the documentation of the respective 
    libraries: `plotly <https://plotly.com/python-api-reference/>`_, `matplotlib <https://matplotlib.org/stable/api/matplotlib_configuration_api.html>`_ and `highcharts <https://api.highcharts.com/highcharts/>`_.

Colors
~~~~~~

.. tab:: Plotly

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")

    **Custom colors**

    .. code-block:: python
        
        data.outliers_plot(
            columns = ["x", "y"], 
            color = "green", 
            outliers_color = "red",
            inliers_color = "pink",
            inliers_border_color = "yellow"
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        fig = data.outliers_plot(
            columns = ["x", "y"], 
            color = "green", 
            outliers_color = "red",
            inliers_color = "pink",
            inliers_border_color = "yellow"
        )
        fig.write_html("figures/plotting_plotly_outliers_2d_plot_custom_color_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_outliers_2d_plot_custom_color_1.html

.. tab:: Highcharts

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")

    **Custom colors**

    .. code-block:: python
        
        data.outliers_plot(
            columns = ["x", "y"], 
            color = "green", 
            outliers_color = "red",
            inliers_color = "pink",
            inliers_border_color = "yellow"
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        fig = data.outliers_plot(
            columns = ["x", "y"], 
            color = "green", 
            outliers_color = "red",
            inliers_color = "pink",
            inliers_border_color = "yellow"
        )
        html_text = fig.htmlcontent.replace("container", "plotting_highcharts_outliers_2d_plot_custom_color_1")
        with open("figures/plotting_highcharts_outliers_2d_plot_custom_color_1.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_highcharts_outliers_2d_plot_custom_color_1.html

.. tab:: Matplolib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    **Custom colors**

    .. ipython:: python
        :okwarning:

        @savefig plotting_matplotlib_outliers_2d_plot_custom_color_1.png
        data.outliers_plot(
            columns = ["x", "y"], 
            color = "green", 
            outliers_color = "red",
            inliers_color = "pink",
            inliers_border_color = "yellow"
        )

____

Size
~~~~

.. tab:: Plotly

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")

    **Custom Width and Height**

    .. code-block:: python
        
        data.outliers_plot(columns = ["x", "y"], width = 300, height = 300)

    .. ipython:: python
        :suppress:
        :okwarning:

        fig = data.outliers_plot(columns = ["x", "y"], width = 300, height = 300)
        fig.write_html("figures/plotting_plotly_outliers_2d_plot_custom_size.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_outliers_2d_plot_custom_size.html

.. tab:: Highcharts

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")

    **Custom Width and Height**

    .. code-block:: python
        
        data.outliers_plot(columns = ["x", "y"], width = 500, height = 200)

    .. ipython:: python
        :suppress:
        :okwarning:

        fig = data.outliers_plot(columns = ["x", "y"], width = 500, height = 200)
        html_text = fig.htmlcontent.replace("container", "plotting_highcharts_outliers_2d_plot_custom_size")
        with open("figures/plotting_highcharts_outliers_2d_plot_custom_size.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_highcharts_outliers_2d_plot_custom_size.html

.. tab:: Matplolib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    **Custom Width and Height**

    .. ipython:: python
        :okwarning:

        @savefig plotting_matplotlib_outliers_2d_plot_single_custom_size.png
        data.outliers_plot(columns = ["x", "y"], width = 6, height = 3)

_____


Text
~~~~

.. tab:: Plotly

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")

    **Custom Title**

    .. code-block:: python
        
        data.outliers_plot(columns = ["x", "y"], ).update_layout(title_text = "Custom Title")

    .. ipython:: python
        :suppress:
        :okwarning:

        fig = data.outliers_plot(columns = ["x", "y"], ).update_layout(title_text = "Custom Title")
        fig.write_html("figures/plotting_plotly_outliers_2d_plot_custom_main_title.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_outliers_2d_plot_custom_main_title.html


    **Custom Axis Titles**

    .. code-block:: python
        
        data.outliers_plot(columns = ["x", "y"], yaxis_title = "Custom Y-Axis Title")

    .. ipython:: python
        :suppress:
        :okwarning:

        fig = data.outliers_plot(columns = ["x", "y"], yaxis_title = "Custom Y-Axis Title")
        fig.write_html("figures/plotting_plotly_outliers_2d_plot_custom_y_title.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_plotly_outliers_2d_plot_custom_y_title.html

.. tab:: Highcharts

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "highcharts")

    **Custom Title Text**

    .. code-block:: python
        
        data.outliers_plot(columns = ["x", "y"], title = {"text": "Custom Title"})

    .. ipython:: python
        :suppress:
        :okwarning:

        fig = data.outliers_plot(columns = ["x", "y"], title = {"text": "Custom Title"})
        html_text = fig.htmlcontent.replace("container", "plotting_highcharts_outliers_2d_plot_custom_text_title")
        with open("figures/plotting_highcharts_outliers_2d_plot_custom_text_title.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_highcharts_outliers_2d_plot_custom_text_title.html

    **Custom Axis Titles**

    .. code-block:: python
        
        data.outliers_plot(columns = ["x", "y"], xAxis = {"title": {"text": "Custom X-Axis Title"}})

    .. ipython:: python
        :suppress:

        fig = data.outliers_plot(columns = ["x", "y"], xAxis = {"title": {"text": "Custom X-Axis Title"}})
        html_text = fig.htmlcontent.replace("container", "plotting_highcharts_outliers_2d_plot_custom_text_xtitle")
        with open("figures/plotting_highcharts_outliers_2d_plot_custom_text_xtitle.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/plotting_highcharts_outliers_2d_plot_custom_text_xtitle.html

.. tab:: Matplolib

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "matplotlib")

    **Custom Title Text**

    .. ipython:: python
        :okwarning:

        @savefig plotting_matplotlib_outliers_2d_plot_custom_title_label.png
        data.outliers_plot(columns = ["x", "y"], ).set_title("Custom Title")

    **Custom Axis Titles**

    .. ipython:: python
        :okwarning:

        @savefig plotting_matplotlib_outliers_2d_plot_custom_yaxis_label.png
        data.outliers_plot(columns = ["x", "y"], ).set_ylabel("Custom Y Axis")

_____

