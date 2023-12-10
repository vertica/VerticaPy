.. _benchmarks.xgboost:


=======
XGBoost
=======

Amazon Redshift | Python | Dask | PySpark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

    **Vertica Version:** 23.3.0-5

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

Higgs Boson
------------

Train: 10.M
Test: 500K

Number of columns: 29

Datatypes of data: Float

Number of feature columns: 28

Amazon
-------

Train: 20,210,579
Test: 5,052,646

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

Test Environment details
^^^^^^^^^^^^^^^^^^^^^^^^^


Vertica
----------


**Parameters:**
- Version: 23.3.0-5
- PlannedConcurrency (general pool): 72
- Memory budget for each query (general pool): ~10GB

Amazon Redshift
----------------

**Parameters:**

.. list-table:: 
    :header-rows: 1

    * - Version
      - Instance Type
      - Cluster
      - vCPU(per node)
      - Memory(per node)
      - Deploy Mode
    * - ???
      - ra3.16xlarge
      - 4 node
      - 48
      - 384
      - N/A


Amazon Sagemaker
------------------

**Parameters:**

.. list-table:: 
    :header-rows: 1

    * - Version
      - Instance Type
      - Cluster
      - vCPU(per node)
      - Memory(per node)
      - Deploy Mode
    * - ???
      - ml.m5.24xlarge
      - 1 node
      - 96
      - 384
      - N/A

But for **1 Billion rows** we have a different configuraiton:

.. list-table:: 
    :header-rows: 1

    * - Version
      - Instance Type
      - Cluster
      - vCPU(per node)
      - Memory(per node)
      - Deploy Mode
    * - ???
      - ml.m5.24xlarge
      - 3 nodes
      - 96
      - 384
      - N/A


Python
---------
**Parameters:**

.. list-table:: 
    :header-rows: 1

    * - Version
      - Instance Type
      - Cluster
      - vCPU(per node)
      - Memory(per node)
      - Deploy Mode
    * - 3.9.15
      - N/A
      - N/A
      - N/A
      - ???
      - N/A



PySPark
--------

**Parameters:**

We have used PySpark Xgboost 1.7.0 version.

.. list-table:: 
    :header-rows: 1

    * - Version
      - Instance Type
      - Cluster
      - vCPU(per node)
      - Memory(per node)
      - Deploy mode
      - Executor Memory
      - Driver Memory
      - Total Executor Cores
    * - 3.3.1
      - ???
      - ???
      - ???
      - ???
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

Below are the results from different dataset sizes. 
Browse throught the tabs to look at each one.

.. tab:: 1 Billion


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
    labels = ['Vertica', 'PySpark']
    heights = [107.45, 1085.84]
    colors = ['blue', 'cyan']
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
        title='Data Size: 1 B',
        #xaxis=dict(title='XGBoost Implementations'),
        yaxis=dict(title='Execution Time (minutes)'),
        bargap=0.2,
        width = 600,
        height = 500
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_1b.html")

  .. raw:: html
    :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_1b.html


.. tab:: 100 Million

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
    labels = ['Vertica', 'Amazon Sagemaker', 'Python', 'PySpark']
    heights = [13.76, 9.11, 5.69, 96.8]
    colors = ['blue', 'orange', 'red', 'cyan']
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
        title='Data Size: 100 M',
        #xaxis=dict(title='XGBoost Implementations'),
        yaxis=dict(title='Execution Time (minutes)'),
        bargap=0.2,
        width = 600,
        height = 500
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_100m.html")

  .. raw:: html
    :file: /project/data/VerticaPy/docs/figures/benchmark_xgboost_higgs_100m.html




.. tab:: 10.5 Million

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
    labels = ['Vertica', 'Amazon Sagemaker', 'Python', 'PySpark']
    heights = [6.1, 2.08, 0.47, 7.26]
    colors = ['blue', 'orange', 'red', 'cyan']
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



Experiments
------------

Below are the results from different experiments. 
Browse throught the tabs to look at each one.

.. tab:: Custom Parameters

  .. csv-table:: Custom Parameters
    :file: /_static/benchmark_xgboost_exp_custom.csv
    :header-rows: 1


  .. ipython:: python
    :suppress:

    import plotly.graph_objects as go
    labels = ['Vertica', 'Amazon Redshift', 'Python', 'PySpark']
    heights = [24.95, 7, 4.33, 56.7]
    colors = ['blue', 'green', 'cyan']
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

.. tab:: Default Parameters

  .. csv-table:: Default Parameters
    :file: /_static/benchmark_xgboost_exp_default.csv
    :header-rows: 2


  .. ipython:: python
    :suppress:

    import plotly.graph_objects as go
    labels = ['Vertica', 'Amazon Redshift', 'Python', 'PySpark']
    heights = [1.27, 8, 3.84, 51.77]
    colors = ['blue', 'green', 'cyan']
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

Below are the results from different experiments of parameters. 
Browse throught the tabs to look at each one.

.. tab:: Custom Parameters

  **Training time Taken**

  .. csv-table:: Custom Parameters
    :file: /_static/benchmark_xgboost_amazon_custom.csv
    :header-rows: 2

  Since the accuracy is similar, we will only show the runtime comparison below:



  .. ipython:: python
    :suppress:

    import plotly.graph_objects as go
    labels = ['Vertica', 'Amazon Redshift', 'Python', 'PySpark']
    heights = [40.53, 7, 9.83, 119.09]
    colors = ['blue', 'green', 'cyan']
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

.. tab:: Default Parameters

  **Training time Taken**

  .. csv-table:: Default Parameters
    :file: /_static/benchmark_xgboost_amazon_default.csv
    :header-rows: 2

  Since the accuracy is similar, we will only show the runtime comparison below:



  .. ipython:: python
    :suppress:

    import plotly.graph_objects as go
    labels = ['Vertica', 'Amazon Redshift', 'Python', 'PySpark']
    heights = [40.53, 7, 9.83, 119.09]
    colors = ['blue', 'green', 'cyan']
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
~~~~~~~~~~~~~~~~~


.. important::

    **Vertica Version:** 11.1.0-0

Dataset
^^^^^^^^

**Amazon**

Size: 25 M

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

.. note::

  In order to get a larger size, we duplicated rows.

Test Environment
^^^^^^^^^^^^^^^^^

Vertica EON
--------------


.. list-table:: 
    :header-rows: 1

    * - Version
      - Instance Type
      - Cluster
      - vCPU(per node)
      - Memory(per node)
      - Deploy Mode
      - OS
      - OS Version
      - Processor freq. (per node)
      - Processor cores (per node) 
      - Type
      - CPU Memory
      - No. of nodes
      - Storage type
    * - 11.1.0-0
      - r4.8xlarge
      - 3 ???
      - ???
      - ???
      - ???
      - ???
      - ???
      - ???
      - ???
      - 32
      - 244
      - 3
      - SSD


Vertica Enterprise
-------------------


.. list-table:: 
    :header-rows: 1

    * - Version
      - Instance Type
      - Cluster
      - vCPU(per node)
      - Memory(per node)
      - Deploy Mode
      - OS
      - OS Version
      - Processor freq. (per node)
      - Processor cores (per node) 
      - Type
      - RAM
    * - 11.1.0-0
      - ???
      - 3 node cluster
      - ???
      - ???
      - ???
      - Red Hat Enterprise Linux 
      - 8.5 (Ootpa)
      - 2.4GHz
      - 4
      - 32
      - 32727072 kB



Comparison
^^^^^^^^^^^

.. list-table:: Time Taken (seconds)
  :header-rows: 1

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
