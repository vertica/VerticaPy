
.. _benchmarks.linear_reg:


==================
Linear Regression
==================

Linear Regression is a fundamental algorithm in machine 
learning and statistics used for predicting a continuous 
outcome variable based on one or more predictor 
variables. It models the relationship between the 
independent variables and the dependent variable by 
fitting a linear equation to observed data. Linear 
Regression is widely employed for tasks such as 
forecasting, risk assessment, and understanding the 
underlying relationships within datasets.

Spark
~~~~~~

.. important::

    **Vertica Version:** ???

This benchmark aims to evaluate the performance of 
Vertica's Linear Regression algorithm in comparison 
to its counterpart in Apache Spark. Through an 
in-depth analysis focusing on speed, accuracy, and 
scalability, we seek to uncover the distinctive 
characteristics of these implementations. By 
shedding light on the strengths and potential 
limitations, this study aims to guide practitioners 
in selecting the most suitable Linear Regression 
solution for their specific use cases.


Dataset
^^^^^^^^

For this dataset, we created an artifical dataset from a Linear Regression model with some noise.

Test Environment
^^^^^^^^^^^^^^^^^^^


.. list-table:: 
  :header-rows: 1

  * - Cluster
    - OS
    - OS Version
    - RAM
    - Processor frequency
    - Processor cores
  * - 3 node cluster
    - Red Hat Enterprise Linux 
    - 8.7 (Ootpa)
    - 755 GB
    - 2.4GHz
    - 36, 2 threads per core

Spark: ``max iter = 100``, ``e = 10^-6``

Vertica: ``max iter = 100``, ``e = 10^-6``


Comparison
^^^^^^^^^^^

.. csv-table:: Vertica vs. Spark
  :file: /_static/benchmark_lr_table.csv
  :header-rows: 2

.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Size': ['1M', '10M'],
      'Vertica BFGS': [4.49, 26.39],
      'Vertica Newton': [4.81, 26.04],
      'Spark BFGS': [1.43, 96.98],
      'Spark Newton': [0.7, 2.09],
  }
  fig = go.Figure()
  bar_width = 0.22  # Set the width of each bar
  fig.add_trace(go.Bar(
      x=data['Size'],
      y=data['Vertica BFGS'],
      width=bar_width,
      text=data['Vertica BFGS'],
      textposition='outside',
      marker_color="black",
      name='Vertica BFGS',
      offset=-0.5
  ))
  fig.add_trace(go.Bar(
      x=data['Size'],
      y=data['Vertica Newton'],
      width=bar_width,
      text=data['Vertica Newton'],
      textposition='outside',
      marker_color="blue",
      name='Vertica Newton',
      offset=-0.25
  ))
  fig.add_trace(go.Bar(
      x=data['Size'],
      y=data['Spark BFGS'],
      width=bar_width,
      text=data['Spark BFGS'],
      textposition='outside',
      marker_color="red",
      name='Spark BFGS',
      offset=0
  ))
  fig.add_trace(go.Bar(
      x=data['Size'],
      y=data['Spark Newton'],
      width=bar_width,
      text=data['Spark Newton'],
      textposition='outside',
      marker_color="green",
      name='Spark Newton',
      offset=0.25
  ))
  fig.update_layout(
      title='Time Comparison (100 Columns)',
      xaxis=dict(title='Size'),
      yaxis=dict(title='Time (seconds)'),
      # barmode='group',
      # bargap=0.8,
      width=600,
      height=500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_linear_regression_spark_time.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_linear_regression_spark_time.html



.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Size': ['100M', '1B'],
      'Vertica BFGS': [84.7, 1748.51],
      'Vertica Newton': [85.93, 1808.56],
      'Spark BFGS': [216, 2568.68],
      'Spark Newton': [68.47, 1788.75],
  }
  fig = go.Figure()
  bar_width = 0.22  # Set the width of each bar
  fig.add_trace(go.Bar(
      x=data['Size'],
      y=data['Vertica BFGS'],
      width=bar_width,
      text=data['Vertica BFGS'],
      textposition='outside',
      marker_color="black",
      name='Vertica BFGS',
      offset=-0.5
  ))
  fig.add_trace(go.Bar(
      x=data['Size'],
      y=data['Vertica Newton'],
      width=bar_width,
      text=data['Vertica Newton'],
      textposition='outside',
      marker_color="blue",
      name='Vertica Newton',
      offset=-0.25
  ))
  fig.add_trace(go.Bar(
      x=data['Size'],
      y=data['Spark BFGS'],
      width=bar_width,
      text=data['Spark BFGS'],
      textposition='outside',
      marker_color="red",
      name='Spark BFGS',
      offset=0
  ))
  fig.add_trace(go.Bar(
      x=data['Size'],
      y=data['Spark Newton'],
      width=bar_width,
      text=data['Spark Newton'],
      textposition='outside',
      marker_color="green",
      name='Spark Newton',
      offset=0.25
  ))
  fig.update_layout(
      title='Time Comparison (100 Columns)',
      xaxis=dict(title='Size'),
      yaxis=dict(title='Time (seconds)'),
      width=600,
      height=500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_linear_regression_spark_time_2.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_linear_regression_spark_time_2.html