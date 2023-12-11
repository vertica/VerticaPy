.. _benchmarks:

===============
Benchmarks
===============



ARIMA
~~~~~~

ARIMA (AutoRegressive Integrated Moving Average) models combine the 
abilities of ``AUTOREGRESSOR`` and ``MOVING_AVERAGE`` models by 
making future predictions based on both preceding time series 
values and errors of previous predictions. ARIMA models also 
provide the option to apply a differencing operation to the input 
data, which can turn a non-stationary time series into a stationary 
time series.

The aim of this benchmark is to compare Vertica algorithm performance 
against 
`python statsmodels <https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html>`_.

Dataset
^^^^^^^

The dataset was artifically created using a Linear model with some random noise.


Test Environment
^^^^^^^^^^^^^^^^^ 

**Vertica Version:** 23.3.0-5

Below are the machine details on which the tests were carried out:


+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+
| Cluster     | OS                        | OS Version            | RAM (per node)         | Processor freq. (per node) | Processor cores (per node)  |
+=============+===========================+=======================+========================+============================+=============================+
| 4 node      | Red Hat Enterprise Linux  | 8.7 (Ootpa)           | 755 GB                 | 2.3 GHz                    | 36, 2 threads per core      |
+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+

**Parameters:** p = 5, d = 1, q = 1

.. csv-table:: Table Title
  :file: /_static/benchmark_arima.csv
  :header-rows: 3

.. note::

  MSE (Mean Squared Error) for Vertica is from summary table (``GET_MODEL_SUMMARY``).


Training Run Time
^^^^^^^^^^^^^^^^^^

.. ipython:: python
  :suppress:

  import plotly.express as px
  import pandas as pd
  df = pd.DataFrame({
      "Size": ["10K", "100K", "1M", "10M", "100M"],
      "Vertica": [0.022, 0.055, 0.515, 4.775, 157.763],
      "Python": [0.064, 0.745, 8.923, 93.307, 1123.966]
  })
  fig = px.line(df, x="Size", y=["Vertica", "Python"], title="Vertica vs Python Performance",
                labels={"value": "Time (minutes)", "variable": "Environment", "Size": "Data Size"},
                line_shape="linear", render_mode="svg")
  fig.update_layout(width = 550)
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_arima_train.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_arima_train.html


Prediction Run Time
^^^^^^^^^^^^^^^^^^^

.. ipython:: python
  :suppress:

  import plotly.express as px
  import pandas as pd
  df = pd.DataFrame({
      "Size": ["10K", "100K", "1M", "10M", "100M"],
      "Vertica": [0.028, 0.056, 0.364, 3.785, 57.052],
      "Python": [0.006, 0.019, 0.027, 0.333, 5.422]
  })

  fig = px.line(df, x="Size", y=["Vertica", "Python"], title="Vertica vs Python Performance",
                labels={"value": "Time (minutes)", "variable": "Environment", "Size": "Data Size"},
                line_shape="linear", render_mode="svg")
  fig.update_layout(width = 550)
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_arima_prediction.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_arima_prediction.html



Mean Squared Error
^^^^^^^^^^^^^^^^^^^

.. ipython:: python
  :suppress:

  import plotly.express as px
  import pandas as pd
  df = pd.DataFrame({
      "Size": ["10K", "100K", "1M", "10M", "100M"],
      "Vertica": [24.54, 30.53, 27.94, 28.52, 32.66],
      "Python": [24.6, 24.97, 25, 24.99, 24.99]
  })

  fig = px.line(df, x="Size", y=["Vertica", "Python"], title="Vertica vs Python Performance",
                labels={"value": "Time (minutes)", "variable": "Environment", "Size": "Data Size"},
                line_shape="linear", render_mode="svg")
  fig.update_layout(width = 550)
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_arima_mse.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_arima_mse.html


----

XGBoost
~~~~~~~~

XGBoost is a highly optimized distributed gradient boosting library 
renowned for its efficiency, flexibility, and portability. Operating 
within the Gradient Boosting framework, XGBoost implements powerful 
machine learning algorithms, specifically designed for optimal 
performance.

This benchmark aims to assess the performance of Vertica's XGBoost 
algorithm in comparison to various XGBoost implementations, 
including those in Spark, Dask, Redshift, and Python.

Implementations to consider:

- Amazon Redshift
- Python
- Dask
- PySpark

By conducting this benchmark, we seek to gain insights into the 
comparative strengths and weaknesses of these implementations. 
Our evaluation will focus on factors such as speed, accuracy, 
and scalability. The results of this study will contribute to a 
better understanding of the suitability of Vertica's XGBoost 
algorithm for diverse data science applications.


Below are the machine details on which the tests were carried out:


+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+
| Cluster     | OS                        | OS Version            | RAM (per node)         | Processor freq. (per node) | Processor cores (per node)  |
+=============+===========================+=======================+========================+============================+=============================+
| 4 node      | Red Hat Enterprise Linux  | 8.7 (Ootpa)           | 755 GB                 | 2.3 GHz                    | 36, 2 threads per core      |
+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+


Datasets
^^^^^^^^^

- Higgs Boson 
- Amazon

**Higgs Boson**

Train: 10.M
Test: 500K

Number of columns: 29

Datatypes of data: Float

Number of feature columns: 28

**Amazon**

Train: 20,210,579
Test: 5,052,646

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

Test Environment details
^^^^^^^^^^^^^^^^^^^^^^^^^


- Vertica version 1

  **Parameters:**

  - Version: 12.0.4-20230103
  - PlannedConcurrency (general pool): 72
  - Memory budget for each query (general pool): ~10GB

- Vertica version 2

  **Parameters:**

  - Version: 23.4 (with VER-88416 added)
  - PlannedConcurrency (general pool): 72
  - Memory budget for each query (general pool): ~10GB

- Amazon Redshift

  **Parameters:**

  .. list-table:: 
   :header-rows: 1

   * - Instance Type
     - Cluster
     - vCPU(per node)
     - Memory(per node)
   * - ra3.16xlarge
     - 4 node
     - 48
     - 384


- Amazon Sagemaker

  **Parameters:**

  .. list-table:: 
   :header-rows: 1

   * - Instance Type
     - Cluster
     - vCPU(per node)
     - Memory(per node)
   * - ml.m5.24xlarge
     - 3 nodes
     - 96
     - 384

  But for **1 Billion rows** we have a different configuraiton:



- Python

  **Parameters:**

  .. list-table:: 
   :header-rows: 1

   * - Version
   * - 3.9.15

- Dask

  **Parameters:**

  .. list-table:: 
   :header-rows: 1

   * - Version
   * - 2022.12.1

- PySPark

  **Parameters:**

  We have used PySpark Xgboost 1.7.0 version.

  .. list-table:: 
   :header-rows: 1

   * - Version
     - Deploy mode
     - Executor Memory
     - Driver Memory
     - Total Executor Cores
   * - 3.3.1
     - client
     - 70GB
     - 50GB
     - 36 ( Per Worker)


Higgs Boson dataset analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Parameters:**
- Number of trees: 10, 
- tree depth=10, 
- number of bins=150


10.5 Million Rows
------------------

.. csv-table:: 10.5 M Rows
  :file: /_static/benchmark_xgboost.csv
  :header-rows: 2

Since the accuracy is similar, we will only show the runtime comparison below:

.. important::

  Amason Redshift is only considering a sample data of size 33,617 for training.

.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  labels = ['Vertica v1', 'Vertica v2', 'Amazon Sagemaker', 'Python', 'Dask', 'PySpark']
  heights = [24.93, 6.1, 2.08, 0.47, 0.56, 7.26]
  colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
  fig = go.Figure()
  for label, height, color in zip(labels, heights, colors):
      fig.add_trace(go.Bar(
          x=[label],
          y=[height],
          marker_color=color,
          text=[height],
          textposition='outside',
          name=label,
      ))
  fig.update_layout(
      title='Data Size: 10.5M',
      #xaxis=dict(title='XGBoost Implementations'),
      yaxis=dict(title='Execution Time (minutes)'),
      bargap=0.2,
      width = 600,
      height = 500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_10m.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_10m.html



100 Million Rows
------------------

.. csv-table:: 100 M Rows
  :file: /_static/benchmark_xgboost_100m.csv
  :header-rows: 2

Since the accuracy is similar, we will only show the runtime comparison below:

.. important::

  Amason Redshift is only considering a sample data of size 33,617 for training.

.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  labels = ['Vertica v1', 'Vertica v2', 'Amazon Sagemaker', 'Python', 'Dask', 'PySpark']
  heights = [32.5, 13.76, 9.11, 5.69, 3.41, 96.8]
  colors = ['blue', 'green', 'orange', 'red', 'purple', 'cyan']
  fig = go.Figure()
  for label, height, color in zip(labels, heights, colors):
      fig.add_trace(go.Bar(
          x=[label],
          y=[height],
          marker_color=color,
          text=[height],
          textposition='outside',
          name=label,
      ))
  fig.update_layout(
      title='Data Size: 10.5M',
      #xaxis=dict(title='XGBoost Implementations'),
      yaxis=dict(title='Execution Time (minutes)'),
      bargap=0.2,
      width = 600,
      height = 500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_100m.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_100m.html



1 Billion Rows
------------------

.. csv-table:: 1 B Rows
  :file: /_static/benchmark_xgboost_1b.csv
  :header-rows: 2

Since the accuracy is similar, we will only show the runtime comparison below:

.. important::

  Amason Redshift is only considering a sample data of size 33,617 for training.

.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  labels = ['Vertica v1', 'Vertica v2', 'Dask', 'PySpark']
  heights = [219.12, 107.45, 29.97, 1085.84]
  colors = ['blue', 'green', 'purple', 'cyan']
  fig = go.Figure()
  for label, height, color in zip(labels, heights, colors):
      fig.add_trace(go.Bar(
          x=[label],
          y=[height],
          marker_color=color,
          text=[height],
          textposition='outside',
          name=label,
      ))
  fig.update_layout(
      title='Data Size: 10.5M',
      #xaxis=dict(title='XGBoost Implementations'),
      yaxis=dict(title='Execution Time (minutes)'),
      bargap=0.2,
      width = 600,
      height = 500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_1b.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_1b.html


Experiments
------------

**Custom Parameters**


.. csv-table:: Custom Parameters
  :file: /_static/benchmark_xgboost_exp_custom.csv
  :header-rows: 1


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  labels = ['Vertica', 'Amazon Redshift', 'Python', 'Dask', 'PySpark']
  heights = [24.95, 7, 4.33, 0.56, 56.7]
  colors = ['blue', 'green', 'purple', 'cyan']
  fig = go.Figure()
  for label, height, color in zip(labels, heights, colors):
      fig.add_trace(go.Bar(
          x=[label],
          y=[height],
          marker_color=color,
          text=[height],
          textposition='outside',
          name=label,
      ))
  fig.update_layout(
      title='Data Size: 10.5M',
      #xaxis=dict(title='XGBoost Implementations'),
      yaxis=dict(title='Execution Time (minutes)'),
      bargap=0.2,
      width = 600,
      height = 500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_exp_custom.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_exp_custom.html


**Default Parameters**


.. csv-table:: Default Parameters
  :file: /_static/benchmark_xgboost_exp_default.csv
  :header-rows: 2


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  labels = ['Vertica', 'Amazon Redshift', 'Python', 'Dask', 'PySpark']
  heights = [1.27, 8, 3.84, 0.45, 51.77]
  colors = ['blue', 'green', 'purple', 'cyan']
  fig = go.Figure()
  for label, height, color in zip(labels, heights, colors):
      fig.add_trace(go.Bar(
          x=[label],
          y=[height],
          marker_color=color,
          text=[height],
          textposition='outside',
          name=label,
      ))
  fig.update_layout(
      title='Data Size: 10.5M',
      #xaxis=dict(title='XGBoost Implementations'),
      yaxis=dict(title='Execution Time (minutes)'),
      bargap=0.2,
      width = 600,
      height = 500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_exp_custom.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_exp_custom.html


Amazon dataset analysis
^^^^^^^^^^^^^^^^^^^^^^^^


.. important::

  Ask Xiaozhong Zhang about difference in accuracy for Vertica ???


**Training time Taken with Custom Parameters**

.. csv-table:: Custom Parameters
  :file: /_static/benchmark_xgboost_amazon_custom.csv
  :header-rows: 2

Since the accuracy is similar, we will only show the runtime comparison below:



.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  labels = ['Vertica', 'Amazon Redshift', 'Python', 'Dask', 'PySpark']
  heights = [40.53, 7, 9.83, 0.86, 119.09]
  colors = ['blue', 'green', 'purple', 'cyan']
  fig = go.Figure()
  for label, height, color in zip(labels, heights, colors):
    fig.add_trace(go.Bar(
      x=[label],
      y=[height],
      marker_color=color,
      text=[height],
      textposition='outside',
      name=label,
    ))
  fig.update_layout(
    title='Data Size: 10.5M',
    #xaxis=dict(title='XGBoost Implementations'),
    yaxis=dict(title='Execution Time (minutes)'),
    bargap=0.2,
    width = 600,
    height = 500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_amazon_exp_custom.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_amazon_exp_custom.html


**Training time Taken with Default Parameters**

.. csv-table:: Default Parameters
  :file: /_static/benchmark_xgboost_amazon_default.csv
  :header-rows: 2

Since the accuracy is similar, we will only show the runtime comparison below:



.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  labels = ['Vertica', 'Amazon Redshift', 'Python', 'Dask', 'PySpark']
  heights = [40.53, 7, 9.83, 0.86, 119.09]
  colors = ['blue', 'green', 'purple', 'cyan']
  fig = go.Figure()
  for label, height, color in zip(labels, heights, colors):
    fig.add_trace(go.Bar(
      x=[label],
      y=[height],
      marker_color=color,
      text=[height],
      textposition='outside',
      name=label,
    ))
  fig.update_layout(
    title='Data Size: 10.5M',
    #xaxis=dict(title='XGBoost Implementations'),
    yaxis=dict(title='Execution Time (minutes)'),
    bargap=0.2,
    width = 600,
    height = 500
  )
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_amazon_exp_default.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_amazon_exp_default.html


Google Big Query
^^^^^^^^^^^^^^^^^


Dataset
-------

**Amazon**

Size: 25 M

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

.. note::

  In order to get a larger size, we duplicated rows.

Test Environment
-----------------

- Vertica EON

  .. list-table:: 
    :header-rows: 1

    * - Instance
      - Type
      - CPU Memory
      - No. of nodes
      - Storage type
    * - r4.8xlarge
      - 32
      - 244
      - 3
      - SSD


- Vertica Enterprise

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
-----------

.. list-table:: Time Taken (seconds)
  :header-rows: 2

  * - Metrics
    - Vertica EON
    - Google BQ
    - Vertica Enterprise
  * - Training
    - 1381.36
    - 1060
    - 1260.09
  * - Predicting (25M)
    - 128.86
    - 19.1
    - 119.83



.. ipython:: python
  :suppress:

  import plotly.graph_objects as go

  labels = ['Vertica EON', 'Vertica Enterprise', 'Google BQ']
  train_times = [1381.36, 1260.09, 1060]
  predict_times = [128.86, 119.83, 19.1]
  colors = ['blue', 'green', 'purple']
  fig = go.Figure()
  bar_width = 0.3  # Set the width of each bar
  gap_width = -0.1  # Set the gap width between bars
  fig.add_trace(
    go.Bar(
      x=[label for label in labels],
      y=train_times,
      width=bar_width,
      marker_color=colors,
      text=train_times,
      textposition='outside',
      name=f'Training',
    )
  )
  fig.add_trace(go.Bar(x=[label for label in labels],y=predict_times,width=bar_width,marker_color=colors,text=predict_times,textposition='outside',name=f'Predicting',offset=bar_width + gap_width,))
  fig.update_layout(title='Training & Predicting', yaxis=dict(title='Execution Time (seconds)'), barmode='group',bargap=0.2,width=600,height=500,)
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_google_bq.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_google_bq.html

----

Random Forest
~~~~~~~~~~~~~~

Madlib
^^^^^^^

Comparison with the `Madlib Random Forest model <https://madlib.apache.org/docs/v1.10/group__grp__random__forest.html>`_.

Dataset
--------


**Amazon**

Train: 20,210,579
Test: 5,052,646

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

.. note::

  In order to get a larger size, we duplicated rows.

Test Environment
-----------------

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
-----------

It was running for more than 11 hours so the test was abondoned.

Spark ML
^^^^^^^^^

Dataset
--------


**Amazon**

Size: 25 M

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

.. note::

  In order to get a larger size, we duplicated rows.

Test Environemnt
-----------------

**Single Node**

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


**Multi Node**

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
-----------

**Single Node**

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


**Multi Node**


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



Naive Bayes Multinomial
~~~~~~~~~~~~~~~~~~~~~~~~

The purpose is to compare the `Spark Naive Bayes Algorithm <\\wsl.localhost\Ubuntu\home\ughumman\VerticaPyLab\project\data\VerticaPy\docs\source\benchmarks.rst>`_.

Dataset
^^^^^^^^

**Amazon**

Size: 25 M

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

Test Environment
^^^^^^^^^^^^^^^^^^^



**Single Node**

.. important::

  Tarun - confirm the machine specs! Inlcuding Spark-submit params.

+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+
| Cluster     | OS                        | OS Version            | RAM (per node)         | Processor freq. (per node) | Processor cores (per node)  |
+=============+===========================+=======================+========================+============================+=============================+
| 2 nodes     | Red Hat Enterprise Linux  | 8.3 (Ootpa)           | 32728552 kB            | 2.4 GHz                    | 4                           |
+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+


**Multi Node**

.. important::

  Are processor cores PER NODE?

+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+
| Cluster     | OS                        | OS Version            | RAM (per node)         | Processor freq. (per node) | Processor cores (per node)  |
+=============+===========================+=======================+========================+============================+=============================+
| 2 nodes     | Red Hat Enterprise Linux  | 8.3 (Ootpa)           | 32728552 kB            | 2.4 GHz                    | 4                           |
+-------------+---------------------------+-----------------------+------------------------+----------------------------+-----------------------------+


Comparison
^^^^^^^^^^^


**Single Node**

.. list-table:: Time in secs
  :header-rows: 1

  * - 
    - Training
    - Prediction - 25 M
    - Accuracy
    - AUC
  * - Spark
    - 145.7
    - 1095.79
    - 150.55
    - 146.58
  * - Vertica
    - 9.08
    - 207.56
    - 0.99
    - 2.19


.. list-table:: 
  :header-rows: 1

  * - Metrics
    - Vertica
    - Spark
  * - Accuracy
    - 0.85
    - 0.85
  * - AUC
    - 0.85
    - 0.77


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Metric': ['Train model', 'Prediction', 'Accuracy', 'AUC'],
      'Spark': [145.70, 1095.79, 150.55, 146.58],
      'Vertica': [9.08, 207.56, 0.99, 2.19]
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
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_naive_bayes_spark_single_time.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_naive_bayes_spark_single_time.html


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Metric': ['Accuracy', 'AUC'],
      'Spark': [0.85, 0.77],
      'Vertica': [0.85, 0.85]
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
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_naive_bayes_spark_single_accuracy.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_naive_bayes_spark_single_accuracy.html


**Multi Node**


.. list-table:: Time in secs
  :header-rows: 1

  * - 
    - Training
    - Prediction- 25 M
    - Accuracy
    - AUC
  * - Spark
    - 69.16
    - 1134.03
    - 64.46
    - 63.70
  * - Vertica
    - 4.83
    - 103.9
    - 0.74
    - 0.78


.. list-table:: 
  :header-rows: 1

  * - Metrics
    - Vertica
    - Spark
  * - Accuracy
    - 0.85
    - 0.85
  * - AUC
    - 0.85
    - 0.77


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Metric': ['Train model', 'Prediction', 'Accuracy', 'AUC'],
      'Spark': [69.16, 1134.03, 64.46, 63.70],
      'Vertica': [4.83, 103.90, 0.74, 0.78]
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
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_naive_bayes_spark_multi_time.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_naive_bayes_spark_multi_time.html


.. ipython:: python
  :suppress:

  import plotly.graph_objects as go
  data = {
      'Metric': ['Accuracy', 'AUC'],
      'Spark': [0.85, 0.77],
      'Vertica': [0.85, 0.85]
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
  fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_naive_bayes_spark_multi_accuracy.html")

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/benchmark_naive_bayes_spark_multi_accuracy.html

Linear Regression
~~~~~~~~~~~~~~~~~~


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

Logistic Regression
~~~~~~~~~~~~~~~~~~~


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



.. toctree::
    :maxdepth: 1
    :titlesonly:

    benchmarks_arima
    benchmarks_xgboost
    benchmarks_naive
    benchmarks_random_forest
    benchmarks_linear_reg
    benchmarks_logistic_reg