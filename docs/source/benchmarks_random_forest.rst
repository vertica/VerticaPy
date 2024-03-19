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


Vertica vs Spark ML
~~~~~~~~~~~~~~~~~~~~

.. important::

  |  *Version Details*
  |  **Vertica:** 11.1.0-0
  |  **Spark:** 3.2.1

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


.. tab:: Amazon


  Size: 25 M

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

.. tab:: Vertica

  .. tab:: Single Node


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
        * - 11.1.0-0
          - On-Premises VM
          - 1 node
          - 8
          - 20393864 kB
          - Enterprise
          - Red Hat Enterprise Linux
          - 7.6 (Maipo)
          - 2.3 GHz

  .. tab:: Multi Node

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
        * - 11.1.0-0
          - On-Premises VM
          - 4 nodes
          - 8
          - 20393864 kB 
          - Enterprise
          - Red Hat Enterprise Linux
          - 7.6 (Maipo)
          - 2.3 GHz


.. tab:: Spark

  .. tab:: Single Node


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
        * - 3.2.1
          - On-Premises VM
          - 1 node
          - 8
          - 20393864 kB
          - NA
          - Red Hat Enterprise Linux
          - 7.6 (Maipo)
          - 2.3 GHz

  .. tab:: Multi Node

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
        * - 3.2.1
          - On-Premises VM
          - 4 nodes
          - 8
          - 20393864 kB 
          - NA
          - Red Hat Enterprise Linux
          - 7.6 (Maipo)
          - 2.3 GHz


Comparison
^^^^^^^^^^^^

.. tab:: Sinlge Node

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

  Browse throught the tabs to see the time and accuracy comparison:

  .. tab:: Time

    .. ipython:: python
      :suppress:
      :okwarning:

      import plotly.graph_objects as go
      data = {
          'Metric': ['Train model', 'Prediction', 'Accuracy', 'AUC'],
          'Spark': [1096, 1581, 248.4, 240.6],
          'Vertica': [650.27, 150.09, 1.24, 1.11]
      }
      fig = go.Figure()
      fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict({"color": "#888888"}),)
      bar_width = 0.22  # Set the width of each bar
      gap_width = 0.00  # Set the gap width between bars
      fig.add_trace(go.Bar(
          x=data['Metric'],
          y=data['Spark'],
          width=bar_width,
          text=data['Spark'],
          textposition='outside',
          marker_color= "#B8B7B6",
          name='Spark'
      ))
      fig.add_trace(go.Bar(
          x=data['Metric'],
          y=data['Vertica'],
          width=bar_width,
          text=data['Vertica'],
          textposition='outside',
          name='Vertica',
          marker_color= "#1A6AFF",
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_random_forest_spark_single_time.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_random_forest_spark_single_time.html

  .. tab:: Accuracy

    .. ipython:: python
      :suppress:
      :okwarning:

      import plotly.graph_objects as go
      data = {
          'Metric': ['Accuracy', 'AUC'],
          'Spark': [0.89, 0.75],
          'Vertica': [0.90, 0.94]
      }
      fig = go.Figure()
      fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict({"color": "#888888"}),)
      bar_width = 0.22  # Set the width of each bar
      gap_width = 0.00  # Set the gap width between bars
      fig.add_trace(go.Bar(
          x=data['Metric'],
          y=data['Spark'],
          width=bar_width,
          text=data['Spark'],
          textposition='outside',
          marker_color= "#B8B7B6",
          name='Spark'
      ))
      fig.add_trace(go.Bar(
          x=data['Metric'],
          y=data['Vertica'],
          width=bar_width,
          text=data['Vertica'],
          textposition='outside',
          name='Vertica',
          marker_color= "#1A6AFF",
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_random_forest_spark_single_accuracy.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_random_forest_spark_single_accuracy.html

.. tab:: Multi Node

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

  Browse throught the tabs to see the time and accuracy comparison:

  .. tab:: Time

    .. ipython:: python
      :suppress:
      :okwarning:

      import plotly.graph_objects as go
      data = {
          'Metric': ['Train model', 'Prediction', 'Accuracy', 'AUC'],
          'Spark': [409.5, 1326.3, 70.72, 66.93],
          'Vertica': [249.64, 69.25, 1.26, 0.43]
      }
      fig = go.Figure()
      fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict({"color": "#888888"}),)
      bar_width = 0.22  # Set the width of each bar
      gap_width = 0.00  # Set the gap width between bars
      fig.add_trace(go.Bar(
          x=data['Metric'],
          y=data['Spark'],
          width=bar_width,
          text=data['Spark'],
          textposition='outside',
          marker_color= "#B8B7B6",
          name='Spark'
      ))
      fig.add_trace(go.Bar(
          x=data['Metric'],
          y=data['Vertica'],
          width=bar_width,
          text=data['Vertica'],
          textposition='outside',
          name='Vertica',
          marker_color= "#1A6AFF",
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_random_forest_spark_multi_time.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_random_forest_spark_multi_time.html

  .. tab:: Accuracy

    .. ipython:: python
      :suppress:
      :okwarning:

      import plotly.graph_objects as go
      data = {
          'Metric': ['Accuracy', 'AUC'],
          'Spark': [0.89, 0.75],
          'Vertica': [0.90, 0.95]
      }
      fig = go.Figure()
      fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict({"color": "#888888"}),)
      bar_width = 0.22  # Set the width of each bar
      gap_width = 0.00  # Set the gap width between bars
      fig.add_trace(go.Bar(
          x=data['Metric'],
          y=data['Spark'],
          width=bar_width,
          text=data['Spark'],
          textposition='outside',
          marker_color= "#B8B7B6",
          name='Spark'
      ))
      fig.add_trace(go.Bar(
          x=data['Metric'],
          y=data['Vertica'],
          width=bar_width,
          text=data['Vertica'],
          textposition='outside',
          name='Vertica',
          marker_color= "#1A6AFF",
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_random_forest_spark_multi_accuracy.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_random_forest_spark_multi_accuracy.html


Vertica vs Madlib
~~~~~~~~~~~~~~~~~~

.. important::

    **Vertica Version:** 23.3.0-5

Comparison with the `Madlib Random Forest model <https://madlib.apache.org/docs/v1.10/group__grp__random__forest.html>`_.

Dataset
^^^^^^^^


.. tab:: Amazon

  .. ipython:: python
    :suppress:
    :okwarning:

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
    fig.write_html("SPHINX_DIRECTORY/figures/benchmark_rf_amazon_data.html")

  .. raw:: html
    :file: SPHINX_DIRECTORY/figures/benchmark_rf_amazon_data.html



  .. list-table:: 
      :header-rows: 1

      * - No. of Columns
      * - 106
      
  Datatypes of data: :bdg-primary-line:`Float`

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

.. important::

  Since all **Madlib** runs were failing for 
  this size of dataset so the benchmark was abandoned.