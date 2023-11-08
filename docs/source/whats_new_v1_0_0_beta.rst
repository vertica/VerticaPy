.. _whats_new_v1_0_0:

===============
Version 1.0.0
===============



This is the first release of VerticaPy. We have fixed some bugs to make it even more stable, and we have also added a lot of new functionalities for our users that we thought were essential.

This release contains some major changes, including:

- Requirements update: Python version 3.13 is now supported.

.. note::

  An internal minimum function python decorator (@check_minimum_version) warns users if any of the Vertica modules do not meet the requirement for the function in use.

  
Bug fixes
----------
- AutoML Error: An error prompt is now displayed when no model fits.
- Cramer's V calculation is now fixed.
- Colors can now be changeg correctly for Matplotlib - Candlestick plot 
- Isolation Forest Anomaly plot is now fixed.
- Plotly LOF 3D plot is fixed.
- Graphviz tree plot display is fixed.




Machine Learning Support
--------------------------
- VerticaPy now supports these Time-Series Vertica models:
  - AutoRegressive (AR)
  - MovingAverages (MA)
  - AutoRegressive Moving Averages (ARMA)
  - AutoRegressive Integrated Moving Averages (ARIMA)
- Now we also support Vertica Poisson Regression
- Model Tracking and Versioning now supported.
  Check out :ref:`notebooks/ml/model_tracking_versioning/index.ipynb` for more details.
- Model Export and Import:
  Now models can be exported to ``pmml``, ``tensorflow``, and ``binary``. 
- AUC now has more averaging options for multi-class dataset:
  - ``weighted``
  - ``score``
  - ``none``


Plotting
----------
- Plotly Outliers plot now has the option to customize colors using the ``colors`` parameter.
- Plotly Voronoi plot colors can also be changed.
- Plotly LOF plot colors can be changed. 
- Validation Curve Plot now has the option to either return the curve or only display results.
- Fixed bounds for Highcharts ACF plot.
- For majority of plots, the colors can be changed by ``colors`` parameter.
- Added Plotly line plots: area, stacked, and fully-stacked.
- Plotly Contout plot colors can be modified.
- Plotly Range plot
  - Can draw multiple plots.
  - Color change is very easy with ``colors`` = ``List`` option e.g.

  .. code-block:: python

    fig=data.range_plot(["col1","col2"],ts = "date", plot_median=True,
      colors=["black","yellow"]
    )

- Plotly Scatter plot now has the option to plot Bubble plot.
- Plotly Pie chart now has the option to change color and size.
- Highcharts Histogram plot is now available.
- PLotly Histogram plot now allows multiple plots.


Miscellaneous
-------------

- Docstrings have been enriched to add examples and other details that will help in creating a more helpful doc.
- A new dataset "Africa Education" has been added to the dataset library. It can be easily imported using:

.. code-block:: python

  from verticapy.datasets import load_africa_education

- Now we use the DISTRIBUTED_SEEDED_RANDOM function instead of SEEDED_RANDOM in Vertica versions higher than 23.
- Some new functions taht help in viewing and using nested data:
  - ``explode_array`` is a ``vDataFrame`` function that allows users to expand the contents of a nested column.

Internal
---------

- More coverage of Unit Tests based on the new UT format.