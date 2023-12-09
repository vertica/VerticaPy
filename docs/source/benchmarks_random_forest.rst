.. _benchmarks.random_forest:


==============
Random Forest
==============


Random Forest is a versatile ensemble learning method that 
excels in making predictions across various domains, 
including classification and regression tasks. It operates 
by constructing a multitude of decision trees during 
training and outputs the mode or mean prediction of the 
individual trees for classification or regression, 
respectively. Renowned for its robustness and resistance 
to overfitting, Random Forest mitigates the shortcomings of 
individual decision trees by leveraging the diversity of an 
ensemble.


Spark ML
~~~~~~~~~

.. important::

    **Vertica Version:** ???

In this benchmark, we aim to evaluate the performance of 
Vertica's Random Forest algorithm in comparison to the 
implementation in Apache Spark. Focusing on the crucial 
aspects of speed, accuracy, and scalability, our analysis 
seeks to provide valuable insights into the strengths and 
limitations of these two implementations. The comparative 
study will contribute to a nuanced understanding of the 
suitability of Vertica's Random Forest algorithm for diverse 
data science applications, particularly when pitted against 
the well-established capabilities of Spark.

Dataset
^^^^^^^^


**Amazon**

Size: 25 M

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

.. note::

  In order to get a larger size, we duplicated rows.

Test Environemnt
^^^^^^^^^^^^^^^^^

Single Node
------------

.. important::

  Tarun - confirm the machine specs! Inlcuding Spark-submit params.

+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+
| Cluster     | OS                        | OS Version            | RAM (per node)         | Processor freq. (per node) | Processor cores (per node)  |
+=============+===========================+=======================+========================+============================+=============================+
| 2 nodes     | Red Hat Enterprise Linux  | 8.3 (Ootpa)           | 32728552 kB            | 2.4 GHz                    | 4                           |
+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+

**Spark-submit parameters:**

.. list-table:: 
  :header-rows: 1

  * - Deploy mode
    - Executor memory
    - Driver memory
    - Total executor cores
    - Class
  * - client
    - 20 GB
    - 5 GB
    - 4
    - pyspark.ml.classification


Multi Node
------------

.. important::

  Are processor cores PER NODE?

+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+
| Cluster     | OS                        | OS Version            | RAM (per node)         | Processor freq. (per node) | Processor cores (per node)  |
+=============+===========================+=======================+========================+============================+=============================+
| 2 nodes     | Red Hat Enterprise Linux  | 8.3 (Ootpa)           | 32728552 kB            | 2.4 GHz                    | 4                           |
+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+

**Spark-submit parameters:**

.. list-table:: 
  :header-rows: 1

  * - Deploy mode
    - Executor memory
    - Driver memory
    - Total executor cores
    - Class
  * - client
    - 20 GB
    - 5 GB
    - 4
    - pyspark.ml.classification


Comparison
^^^^^^^^^^^^

Single Node
----------------

.. list-table:: Time in secs
  :header-rows: 1

  * - 
    - Training
    - Prediction - 25 M
    - Accuracy
    - AUC
  * - Spark
    - 1096
    - 1581
    - 248.4
    - 240.6
  * - Vertica
    - 650.27
    - 150.09
    - 1.24
    - 1.11


.. list-table:: 
  :header-rows: 1

  * - Metrics
    - Vertica
    - Spark
  * - Accuracy
    - 0.90
    - 0.89
  * - AUC
    - 0.94
    - 0.75


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Metric': ['Train model', 'Prediction', 'Accuracy', 'AUC'],
      'Spark': [1096, 1581, 248.4, 240.6],
      'Vertica': [650.27, 150.09, 1.24, 1.11]
  }
  fig = go.Figure()
  bar_width = 0.22  # Set the width of each bar
  gap_width = 0.00  # Set the gap width between bars
  fig.add_trace(go.Bar(
      x=data['Metric'],
      y=data['Spark'],
      width=bar_width,
      text=data['Spark'],
      textposition='outside',
      marker_color= "blue",
      name='Spark'
  ))
  fig.add_trace(go.Bar(
      x=data['Metric'],
      y=data['Vertica'],
      width=bar_width,
      text=data['Vertica'],
      textposition='outside',
      name='Vertica',
      marker_color= "black",
      offset=0.15
  ))
  fig.update_layout(
      title='Time Comaprison (Spark vs. Vertica)',
      xaxis=dict(title='Metrics'),
      yaxis=dict(title='Time (seconds)'),
      barmode='group',
      bargap=gap_width,
      width=550,
      height=600
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_random_forest_spark_single_time.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_random_forest_spark_single_time.html


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Metric': ['Accuracy', 'AUC'],
      'Spark': [0.89, 0.75],
      'Vertica': [0.90, 0.94]
  }
  fig = go.Figure()
  bar_width = 0.22  # Set the width of each bar
  gap_width = 0.00  # Set the gap width between bars
  fig.add_trace(go.Bar(
      x=data['Metric'],
      y=data['Spark'],
      width=bar_width,
      text=data['Spark'],
      textposition='outside',
      marker_color= "blue",
      name='Spark'
  ))
  fig.add_trace(go.Bar(
      x=data['Metric'],
      y=data['Vertica'],
      width=bar_width,
      text=data['Vertica'],
      textposition='outside',
      name='Vertica',
      marker_color= "black",
      offset=0.15
  ))
  fig.update_layout(
      title='Accuracy Comaprison (Spark vs. Vertica)',
      xaxis=dict(title='Metrics'),
      yaxis=dict(title='Time (seconds)'),
      barmode='group',
      bargap=gap_width,
      width=550,
      height=600
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_random_forest_spark_single_accuracy.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_random_forest_spark_single_accuracy.html


Multi Node
------------

.. list-table:: Time in secs
  :header-rows: 1

  * - 
    - Training
    - Prediction- 25 M
    - Accuracy
    - AUC
  * - Spark
    - 409.5
    - 1326.3
    - 70.72
    - 66.93
  * - Vertica
    - 249.64
    - 69.25
    - 1.26
    - 0.43


.. list-table:: 
  :header-rows: 1

  * - Metrics
    - Vertica
    - Spark
  * - Accuracy
    - 0.90
    - 0.89
  * - AUC
    - 0.95
    - 0.75


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Metric': ['Train model', 'Prediction', 'Accuracy', 'AUC'],
      'Spark': [409.5, 1326.3, 70.72, 66.93],
      'Vertica': [249.64, 69.25, 1.26, 0.43]
  }
  fig = go.Figure()
  bar_width = 0.22  # Set the width of each bar
  gap_width = 0.00  # Set the gap width between bars
  fig.add_trace(go.Bar(
      x=data['Metric'],
      y=data['Spark'],
      width=bar_width,
      text=data['Spark'],
      textposition='outside',
      marker_color= "blue",
      name='Spark'
  ))
  fig.add_trace(go.Bar(
      x=data['Metric'],
      y=data['Vertica'],
      width=bar_width,
      text=data['Vertica'],
      textposition='outside',
      name='Vertica',
      marker_color= "black",
      offset=0.15
  ))
  fig.update_layout(
      title='Time Comaprison (Spark vs. Vertica)',
      xaxis=dict(title='Metrics'),
      yaxis=dict(title='Time (seconds)'),
      barmode='group',
      bargap=gap_width,
      width=550,
      height=600
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_random_forest_spark_multi_time.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_random_forest_spark_multi_time.html


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Metric': ['Accuracy', 'AUC'],
      'Spark': [0.89, 0.75],
      'Vertica': [0.90, 0.95]
  }
  fig = go.Figure()
  bar_width = 0.22  # Set the width of each bar
  gap_width = 0.00  # Set the gap width between bars
  fig.add_trace(go.Bar(
      x=data['Metric'],
      y=data['Spark'],
      width=bar_width,
      text=data['Spark'],
      textposition='outside',
      marker_color= "blue",
      name='Spark'
  ))
  fig.add_trace(go.Bar(
      x=data['Metric'],
      y=data['Vertica'],
      width=bar_width,
      text=data['Vertica'],
      textposition='outside',
      name='Vertica',
      marker_color= "black",
      offset=0.15
  ))
  fig.update_layout(
      title='Accuracy Comaprison (Spark vs. Vertica)',
      xaxis=dict(title='Metrics'),
      yaxis=dict(title='Time (seconds)'),
      barmode='group',
      bargap=gap_width,
      width=550,
      height=600
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_random_forest_spark_multi_accuracy.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_random_forest_spark_multi_accuracy.html





Madlib
~~~~~~

.. important::

    **Vertica Version:** ???

Comparison with the `Madlib Random Forest model <https://madlib.apache.org/docs/v1.10/group__grp__random__forest.html>`_.

Dataset
^^^^^^^^


**Amazon**

Train: 20,210,579
Test: 5,052,646

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

.. note::

  In order to get a larger size, we duplicated rows.

Test Environment
^^^^^^^^^^^^^^^^^

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
    - 8.5 (Ootpa)
    - 32727072 kB
    - 2.4GHz
    - 4


Comparison
^^^^^^^^^^^

It was running for more than 11 hours so the test was abondoned.