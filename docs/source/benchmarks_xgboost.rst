.. _benchmarks.xgboost:


=======
XGBoost
=======

Vertica vs Amazon Redshift | Python | PySpark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

  |  *Version Details*
  |  **Vertica:** 23.4
  |  **Amazon Redshift:** Jan 2023
  |  **Amazon Sagemaker:** Jan 2023
  |  **Python Native XGBoost:** 3.9.15
  |  **PySark:** 3.3.1

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

.. tab:: Higgs Boson

  .. ipython:: python
    :suppress:

    import plotly.express as px
    import pandas as pd
    training_data_count = 10.5e6
    testing_data_count = 500e3
    data = {'Data': ['Training', 'Testing'], 'Count': [training_data_count, testing_data_count]}
    df = pd.DataFrame(data)
    fig = px.pie(df, values='Count', names='Data', title='Training and Testing Data Distribution', 
      labels={'Count': 'Data Count'}, color_discrete_sequence=['blue', 'black'])
    fig.update_traces(textinfo='value')
    fig.update_layout(width = 550)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict({"color": "#888888"}),)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict({"color": "#888888"}),)
    fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_data.html")

  .. raw:: html
    :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_data.html


  .. list-table:: 
      :header-rows: 1

      * - No. of Columns
      * - 29


  Datatypes of data: :bdg-primary-line:`Float`


.. tab:: Amazon

  .. ipython:: python
    :suppress:

    import plotly.express as px
    import pandas as pd
    training_data_count = 20210579
    testing_data_count = 5052646
    data = {'Data': ['Training', 'Testing'], 'Count': [training_data_count, testing_data_count]}
    df = pd.DataFrame(data)
    fig = px.pie(df, values='Count', names='Data', title='Training and Testing Data Distribution', 
      labels={'Count': 'Data Count'}, color_discrete_sequence=['blue', 'black'])
    fig.update_traces(textinfo='value')
    fig.update_layout(width = 550)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict({"color": "#888888"}),)
    fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_amazon_data.html")

  .. raw:: html
    :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_amazon_data.html



  .. list-table:: 
      :header-rows: 1

      * - No. of Columns
      * - 106

  Datatypes of data: :bdg-primary-line:`Float`


Test Environment details
^^^^^^^^^^^^^^^^^^^^^^^^^

Below are the configurations for each 
algorithm that was tested:

.. tab:: Vertica

  **Parameters:**
  - PlannedConcurrency (general pool): 72
  - Memory budget for each query (general pool): ~10GB

  .. list-table:: 
      :header-rows: 1

      * - Version
        - Instance Type
        - Cluster
        - vCPU (per node)
        - Memory (per node)
        - Deploy Mode
        - OS
        - OS Version
        - Processor freq. (per node)
      * - 23.4
        - On Premise VM
        - 4 node 
        - 36, 2 threads per core
        - 755 GB
        - Enterprise
        - Red Hat Enterprise Linux  
        - 8.7 (Ootpa)   
        - 2.3 GHz  


.. tab:: Amazon Redshift

  **Parameters:**

  .. list-table:: 
      :header-rows: 1

      * - Version
        - Instance Type
        - Cluster
        - vCPU (per node)
        - Memory (per node)
        - Deploy Mode
      * - Jan 2023
        - ra3.16xlarge
        - 4 node
        - 48
        - 384
        - N/A

.. tab:: Amazon Sagemaker

  **Parameters:**

  .. list-table:: 
      :header-rows: 1

      * - Version
        - Instance Type
        - Cluster
        - vCPU (per node)
        - Memory (per node)
        - Deploy Mode
      * - Jan 2023
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
        - vCPU (per node)
        - Memory (per node)
        - Deploy Mode
      * - Jan 2023
        - ml.m5.24xlarge
        - 3 nodes
        - 96
        - 384
        - N/A

.. tab:: Python

  **Parameters:**

  .. list-table:: 
      :header-rows: 1

      * - Version
        - Instance Type
        - Cluster
        - vCPU (per node)
        - Memory (per node)
        - Deploy Mode
      * - 3.9.15
        - N/A
        - N/A
        - N/A
        - N/A
        - N/A


.. tab:: Pyspark

  **Parameters:**

  We have used PySpark Xgboost 1.7.0 version.

  .. list-table:: 
      :header-rows: 1

      * - Version
        - Instance Type
        - Cluster
        - vCPU (per node)
        - Memory (per node)
        - Deploy mode
        - Executor Memory
        - Driver Memory
      * - 3.3.1
        - N/A
        - N/A
        - 36 ( Per Worker)
        - N/A
        - client
        - 70GB
        - 50GB


Parameters
-----------



.. tab:: Custom Parameters

  +------------------+------------+----------------+----------------+-----------------------------------+
  |    Platform      | Num Trees  | Tree Depth     | Number of Bins | Feature Importance (Top 5)        |
  +==================+============+================+================+===================================+
  | Vertica          | 10         | 10             | 150            | col26, col27, col28, col23, col25 |
  +------------------+------------+----------------+----------------+-----------------------------------+
  | Amazon Redshift  | 100        | 10             | 150            | col25, col27, col26, col22, col24 |
  +------------------+------------+----------------+----------------+-----------------------------------+
  | Python           | 10         | 10             | 150            | col26, col28, col27, col23, col6  |
  +------------------+------------+----------------+----------------+-----------------------------------+
  | Dask (Python)    | 10         | 10             | 150            | col26, col28, col27, col23, col6  |
  +------------------+------------+----------------+----------------+-----------------------------------+
  | Spark            | 100        | 10             | 150            | col25, col27, col26, col22, col5  |
  +------------------+------------+----------------+----------------+-----------------------------------+

.. tab:: Default Parameters

  +------------------+------------+----------------+----------------+-----------------------------------+
  |    Platform      | Num Trees  | Tree Depth     | Number of Bins | Feature Importance (Top 5)        |
  +==================+============+================+================+===================================+
  | Vertica          | 10         | 6              | 32             | col26, col27, col28, col23, col25 |
  +------------------+------------+----------------+----------------+-----------------------------------+
  | Amazon Redshift  | 10         | 6              | 256            | col25, col27, col26, col22, col24 |
  +------------------+------------+----------------+----------------+-----------------------------------+
  | Python           | 10         | 6              | 256            | col26, col28, col27, col23, col6  |
  +------------------+------------+----------------+----------------+-----------------------------------+
  | Dask (Python)    | 10         | 6              | 256            | col26, col28, col27, col23, col6  |
  +------------------+------------+----------------+----------------+-----------------------------------+
  | Spark            | 100        | 6              | 256            | col25, col27, col26, col22, col5  |
  +------------------+------------+----------------+----------------+-----------------------------------+



Analysis
^^^^^^^^^^

The comparison analysis on both datasets follows:

.. tab:: Higgs Boson dataset analysis

  **Parameters:**
  - Number of trees: 10, 
  - tree depth=10, 
  - number of bins=150

  Below are the results from different dataset sizes. 
  Browse throught the tabs to look at each one.

  .. tab:: 1 Billion


    .. csv-table:: 1B Rows
      :file: /_static/benchmark_xgboost_1b.csv
      :header-rows: 2

    Since the accuracy is similar, we will only show the runtime comparison below:

    .. important::

      **Amazon Redshift** is only considering a sample data of size 33,617 for training.
      Thus, we have removed it from further analysis.

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Amazon Sagemaker', 'Python', 'PySpark']
      heights = [107.45, 720, 0, 1085.84]
      colors = ["#1A6AFF", "#ee145b", "#f0d917", 'black']
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
          yaxis=dict(title='Execution Time (minutes)'),
          bargap=0.2,
          width=600,
          height=500,
          annotations=[
              dict(
                  x='Amazon Sagemaker',
                  y=720,
                  xref="x",
                  yref="y",
                  text="Did not complete in 720 mins",
                  showarrow=False,
                  arrowhead=7,
                  xshift=0,
                  yshift=30,
                  font=dict(color='red', size=14)
              ),
              dict(
                  x='Python',
                  y=0,
                  xref="x",
                  yref="y",
                  text="Memory Error",
                  showarrow=False,
                  arrowhead=7,
                  xshift=0,
                  yshift=30,
                  font=dict(color='red', size=14)
              )
          ]
      )
      fig.update_layout(
        title='Data Size: 1B',
        #xaxis=dict(title='XGBoost Implementations'),
        yaxis=dict(title='Execution Time (minutes)'),
        bargap=0.2,
        width = 600,
        height = 500
      )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_1b.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_1b.html


  .. tab:: 100 Million


    .. csv-table:: 100 M Rows
      :file: /_static/benchmark_xgboost_100m.csv
      :header-rows: 2

    Since the accuracy is similar, we will only show the runtime comparison below:

    .. important::

      Amazon Redshift is only considering a sample data of size 33,617 for training.

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Amazon Sagemaker', 'Python', 'PySpark']
      heights = [13.76, 9.11, 5.69, 96.8]
      colors = ["#1A6AFF", "#ee145b", "#f0d917", 'black']
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_100m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_100m.html




  .. tab:: 10.5 Million

    .. csv-table:: 10.5 M Rows
      :file: /_static/benchmark_xgboost.csv
      :header-rows: 2

    Since the accuracy is similar, we will only show the runtime comparison below:

    .. important::

      Amazon Redshift is only considering a sample data of size 33,617 for training.

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Amazon Sagemaker', 'Python', 'PySpark']
      heights = [6.1, 2.08, 0.47, 7.26]
      colors = ["#1A6AFF", "#ee145b", "#f0d917", 'black']
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_10m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_10m.html



  Experiments
  ++++++++++++

  Below are the results from different experiments. 
  Browse throught the tabs to look at each one.


  .. tab:: Default Parameters

    .. csv-table:: Default Parameters
      :file: /_static/benchmark_xgboost_exp_default.csv
      :header-rows: 2


    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Amazon Redshift', 'Python', 'PySpark']
      heights = [1.27, 8, 3.84, 51.77]
      colors = ["#1A6AFF", 'green', "#f0d917", 'black']
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_exp_custom.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_exp_custom.html

  .. tab:: Custom Parameters

    .. csv-table:: Custom Parameters
      :file: /_static/benchmark_xgboost_exp_custom.csv
      :header-rows: 1


    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Amazon Redshift', 'Python', 'PySpark']
      heights = [24.95, 7, 4.33, 56.7]
      colors = ["#1A6AFF", 'green', "#f0d917", 'black']
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_exp_custom.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_higgs_exp_custom.html



.. tab:: Amazon dataset analysis


  Below are the results from different experiments of parameters. 
  Browse through the tabs to look at each one.


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
      heights = [6.105, 7, 9.78, 122.08]
      colors = ["#1A6AFF", 'green', "#f0d917", 'black']
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_amazon_exp_default.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_amazon_exp_default.html

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
      colors = ["#1A6AFF", 'green', "#f0d917", 'black']
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_amazon_exp_custom.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_amazon_exp_custom.html



Vertica EON vs Vertica Enterprise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. important::

    **Vertica Version:** 11.1.0-0

Dataset
^^^^^^^^

**Amazon**

.. list-table:: 
    :header-rows: 1

    * - No. of Rows
      - No. of Columns
    * - 25 M
      - 106

Datatypes of data: :bdg-primary-line:`Float`


.. note::

  In order to get a larger size, we duplicated rows.

Test Environment
^^^^^^^^^^^^^^^^^

.. tab:: Vertica EON

  .. list-table:: 
      :header-rows: 1

      * - Version
        - Instance Type
        - Cluster
        - vCPU (per node)
        - Memory (per node)
        - Deploy Mode
        - OS
        - OS Version
        - Processor freq. (per node)
        - Processor cores (per node) 
        - Type
        - No. of nodes
        - Storage type
      * - 11.1.0-0
        - r4.8xlarge
        - 3 nodes
        - N/A
        - 244 GB
        - Eon
        - Red Hat Enterprise Linux 
        - 8.5 (Ootpa)
        - 2.4GHz
        - N/A
        - 32
        - 3
        - SSD

.. tab:: Vertica Enterprise

  .. list-table:: 
      :header-rows: 1

      * - Version
        - Instance Type
        - Cluster
        - vCPU (per node)
        - Memory (per node)
        - Deploy Mode
        - OS
        - OS Version
        - Processor freq. (per node)
        - Processor cores (per node) 
        - Type
      * - 11.1.0-0
        - On Premise VM
        - 3 node cluster
        - N/A
        - 32727072 kB
        - Enterprise
        - Red Hat Enterprise Linux 
        - 8.5 (Ootpa)
        - 2.4GHz
        - 4
        - 32


Comparison
^^^^^^^^^^^

.. list-table:: Time Taken (seconds)
  :header-rows: 1

  * - Metrics
    - Vertica EON
    - Vertica Enterprise
  * - Training
    - 1381.36
    - 1260.09
  * - Predicting (25M)
    - 128.86
    - 119.83

.. tab:: Training Time

  .. ipython:: python
    :suppress:
    :okwarning:

    import plotly.express as px
    ml_tools = ['EON', 'Enterprise']
    training_times = [1381.36, 1260.09] 
    df = pd.DataFrame({'ML Tool': ml_tools, 'Training Time (seconds)': training_times})
    fig = px.bar(df, x='ML Tool', y='Training Time (seconds)', 
      title='Training Time',
      color='ML Tool',
      color_discrete_map={'EON': "#1A6AFF", 'Enterprise': "#ee145b"})
    fig.update_layout(xaxis_title=None)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict({"color": "#888888"}),)
    fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_eon_vs_enterprise_train.html")

  .. raw:: html
    :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_eon_vs_enterprise_train.html


.. tab:: Prediction Time

  .. ipython:: python
    :suppress:
    :okwarning:

    import plotly.express as px
    ml_tools = ['EON', 'Enterprise']
    training_times = [128.86, 119.83] 
    df = pd.DataFrame({'ML Tool': ml_tools, 'Prediction Time (seconds)': training_times})
    fig = px.bar(df, x='ML Tool', y='Prediction Time (seconds)', 
      title='Prediction Time',
      color='ML Tool',
      color_discrete_map={'EON': "#1A6AFF", 'Enterprise': "#ee145b"})
    fig.update_layout(xaxis_title=None)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict({"color": "#888888"}),)
    fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_eon_vs_enterprise_prediction.html")

  .. raw:: html
    :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_eon_vs_enterprise_prediction.html



.. Google Big Query
.. ~~~~~~~~~~~~~~~~~


.. .. important::

..     **Vertica Version:** 11.1.0-0

.. Dataset
.. ^^^^^^^^

.. **Amazon**

.. Size: 25 M

.. Number of columns : 106

.. Datatypes of data: Float

.. Number of feature columns: 105

.. .. note::

..   In order to get a larger size, we duplicated rows.

.. Test Environment
.. ^^^^^^^^^^^^^^^^^

.. Vertica EON
.. --------------


.. .. list-table:: 
..     :header-rows: 1

..     * - Version
..       - Instance Type
..       - Cluster
..       - vCPU (per node)
..       - Memory (per node)
..       - Deploy Mode
..       - OS
..       - OS Version
..       - Processor freq. (per node)
..       - Processor cores (per node) 
..       - Type
..       - CPU Memory
..       - No. of nodes
..       - Storage type
..     * - 11.1.0-0
..       - r4.8xlarge
..       - 3 ???
..       - ???
..       - ???
..       - ???
..       - ???
..       - ???
..       - ???
..       - ???
..       - 32
..       - 244
..       - 3
..       - SSD


.. Vertica Enterprise
.. -------------------


.. .. list-table:: 
..     :header-rows: 1

..     * - Version
..       - Instance Type
..       - Cluster
..       - vCPU (per node)
..       - Memory (per node)
..       - Deploy Mode
..       - OS
..       - OS Version
..       - Processor freq. (per node)
..       - Processor cores (per node) 
..       - Type
..       - RAM
..     * - 11.1.0-0
..       - ???
..       - 3 node cluster
..       - ???
..       - ???
..       - ???
..       - Red Hat Enterprise Linux 
..       - 8.5 (Ootpa)
..       - 2.4GHz
..       - 4
..       - 32
..       - 32727072 kB



.. Comparison
.. ^^^^^^^^^^^

.. .. list-table:: Time Taken (seconds)
..   :header-rows: 1

..   * - Metrics
..     - Vertica EON
..     - Google BQ
..     - Vertica Enterprise
..   * - Training
..     - 1381.36
..     - 1060
..     - 1260.09
..   * - Predicting (25M)
..     - 128.86
..     - 19.1
..     - 119.83



.. .. ipython:: python
..   :suppress:

..   import plotly.graph_objects as go

..   labels = ['Vertica EON', 'Vertica Enterprise', 'Google BQ']
..   train_times = [1381.36, 1260.09, 1060]
..   predict_times = [128.86, 119.83, 19.1]
..   colors = ["#1A6AFF", 'green', 'purple']
..   fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict({"color": "#888888"}),
        )
..   bar_width = 0.3  # Set the width of each bar
..   gap_width = -0.1  # Set the gap width between bars
..   fig.add_trace(
..     go.Bar(
..       x=[label for label in labels],
..       y=train_times,
..       width=bar_width,
..       marker_color=colors,
..       text=train_times,
..       textposition='outside',
..       name=f'Training',
..     )
..   )
..   fig.add_trace(go.Bar(x=[label for label in labels],y=predict_times,width=bar_width,marker_color=colors,text=predict_times,textposition='outside',name=f'Predicting',offset=bar_width + gap_width,))
..   fig.update_layout(title='Training & Predicting', yaxis=dict(title='Execution Time (seconds)'), barmode='group',bargap=0.2,width=600,height=500,)
..   fig.write_html("SPHINX_DIRECTORY/figures/benchmark_xgboost_google_bq.html")

.. .. raw:: html
..   :file: SPHINX_DIRECTORY/figures/benchmark_xgboost_google_bq.html

