.. _benchmarks.naive:


==============
Naive Bayes
==============


Naive Bayes is a probabilistic classification algorithm 
based on Bayes' theorem, which assumes independence 
between features. This simplicity, combined with its 
efficiency and effectiveness, makes Naive Bayes 
particularly well-suited for various classification 
tasks. By calculating the probability of each class 
based on the input features, Naive Bayes provides a 
straightforward yet powerful approach to predictive 
modeling.

Vertica vs Spark
~~~~~~~~~~~~~~~~

.. important::

  |  *Version Details*
  |  **Vertica:** 11.1.0-0
  |  **Spark:** 3.2.1

The goal is to assess the performance of Vertica's 
Naive Bayes algorithm in direct comparison with the 
implementation in Apache Spark. This evaluation will 
focus on critical factors such as speed, accuracy, and 
scalability, providing valuable insights into the 
comparative strengths and limitations of these two 
implementations. Our study aims to enhance the 
understanding of the applicability of Vertica's Naive 
Bayes algorithm in diverse data science scenarios, 
offering practitioners valuable information for making 
informed algorithmic choices.

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

Test Environment
^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^


.. tab:: Single Node

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

  Browse throught the tabs to see the time and accuracy comparison:

  .. tab:: Time
      
    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      data = {
          'Metric': ['Train model', 'Prediction'],
          'Spark': [145.70, 1095.79],
          'Vertica': [9.08, 207.56]
      }
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_naive_bayes_spark_single_time.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_naive_bayes_spark_single_time.html

  .. tab:: Accuracy

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      data = {
          'Metric': ['Accuracy', 'AUC'],
          'Spark': [0.85, 0.77],
          'Vertica': [0.85, 0.85]
      }
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_naive_bayes_spark_single_accuracy.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_naive_bayes_spark_single_accuracy.html


.. tab:: Multi Node

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

  Browse throught the tabs to see the time and accuracy comparison:

  .. tab:: Time
      
    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      data = {
          'Metric': ['Train model', 'Prediction'],
          'Spark': [69.16, 1134.03],
          'Vertica': [4.83, 103.90]
      }
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_naive_bayes_spark_multi_time.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_naive_bayes_spark_multi_time.html

  .. tab:: Accuracy

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      data = {
          'Metric': ['Accuracy', 'AUC'],
          'Spark': [0.85, 0.77],
          'Vertica': [0.85, 0.85]
      }
      fig = go.Figure()
      fig.update_layout(
          paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(0,0,0,0)",
          font=dict({"color": "#888888"}),
      )
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
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_naive_bayes_spark_multi_accuracy.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_naive_bayes_spark_multi_accuracy.html