
.. _benchmarks.logistic_reg:


===================
Logistic Regression
===================


Logistic Regression is a powerful algorithm employed 
for binary classification tasks. Despite its name, 
it is primarily used for classification rather than 
regression. Logistic Regression models the 
probability that a given instance belongs to a 
particular category and is widely utilized in various 
fields, including healthcare, finance, and marketing. 
Its simplicity, interpretability, and effectiveness 
make it a popular choice for predictive modeling.

Spark
~~~~~~

.. important::

    **Vertica Version:** ???

In this benchmark, we strive to assess the performance 
of Vertica's Logistic Regression algorithm in 
comparison to its implementation in Apache Spark. 
Our evaluation will delve into crucial metrics such as
speed, accuracy, and scalability, aiming to elucidate 
the strengths and potential trade-offs associated with 
these implementations. The results of this study will 
contribute valuable insights for practitioners seeking 
to leverage Logistic Regression for classification 
tasks within diverse data science applications.

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





Comparison
^^^^^^^^^^^


.. csv-table:: Vertica vs. Spark
  :file: /_static/benchmark_logr_table.csv
  :header-rows: 2

.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Size': ['1M', '10M'],
      'Vertica BFGS': [14.74, 45.15],
      'Vertica Newton': [6.7, 28.98],
      'Spark': [4.52, 12.05],
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
      y=data['Spark'],
      width=bar_width,
      text=data['Spark'],
      textposition='outside',
      marker_color="red",
      name='Spark',
      offset=0
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
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_logistic_regression_spark_time.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_logistic_regression_spark_time.html



.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Size': ['100M', '1B'],
      'Vertica BFGS': [36.54, 388.89],
      'Vertica Newton': [194.5, 2389],
      'Spark': [367.27, 2222],
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
      y=data['Spark'],
      width=bar_width,
      text=data['Spark'],
      textposition='outside',
      marker_color="red",
      name='Spark',
      offset=0
  ))
  fig.update_layout(
      title='Time Comparison (100 Columns)',
      xaxis=dict(title='Size'),
      yaxis=dict(title='Time (seconds)'),
      width=600,
      height=500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_logistic_regression_spark_time_2.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_logistic_regression_spark_time_2.html

