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

Spark
~~~~~~

.. important::

  |  *Version Details*
  |  **Vertica:** 23.3.0-5
  |  **Spark:** ???

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

**Amazon**

Size: 25 M

Number of columns : 106

Datatypes of data: Float

Number of feature columns: 105

Test Environment
^^^^^^^^^^^^^^^^^^^



Single Node
-------------

.. important::

  Tarun - confirm the machine specs! Inlcuding Spark-submit params.


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
    * - 11.1.0-0
      - ???
      - 2 nodes
      - ???
      - 32728552 kB 
      - ???
      - Red Hat Enterprise Linux
      - 8.3 (Ootpa)   
      - 2.4 GHz
      - 4

Multi Node
-----------

.. important::

  Are processor cores PER NODE?


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
    * - 11.1.0-0
      - ???
      - 2 nodes
      - ???
      - 32728552 kB 
      - ???
      - Red Hat Enterprise Linux
      - 8.3 (Ootpa)   
      - 2.4 GHz
      - 4

Comparison
^^^^^^^^^^^


Single Node
------------

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
        marker_color= "black",
        name='Spark'
    ))
    fig.add_trace(go.Bar(
        x=data['Metric'],
        y=data['Vertica'],
        width=bar_width,
        text=data['Vertica'],
        textposition='outside',
        name='Vertica',
        marker_color= "blue",
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
    bar_width = 0.22  # Set the width of each bar
    gap_width = 0.00  # Set the gap width between bars
    fig.add_trace(go.Bar(
        x=data['Metric'],
        y=data['Spark'],
        width=bar_width,
        text=data['Spark'],
        textposition='outside',
        marker_color= "black",
        name='Spark'
    ))
    fig.add_trace(go.Bar(
        x=data['Metric'],
        y=data['Vertica'],
        width=bar_width,
        text=data['Vertica'],
        textposition='outside',
        name='Vertica',
        marker_color= "blue",
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


Multi Node
-----------

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
        marker_color= "black",
        name='Spark'
    ))
    fig.add_trace(go.Bar(
        x=data['Metric'],
        y=data['Vertica'],
        width=bar_width,
        text=data['Vertica'],
        textposition='outside',
        name='Vertica',
        marker_color= "blue",
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
    bar_width = 0.22  # Set the width of each bar
    gap_width = 0.00  # Set the gap width between bars
    fig.add_trace(go.Bar(
        x=data['Metric'],
        y=data['Spark'],
        width=bar_width,
        text=data['Spark'],
        textposition='outside',
        marker_color= "black",
        name='Spark'
    ))
    fig.add_trace(go.Bar(
        x=data['Metric'],
        y=data['Vertica'],
        width=bar_width,
        text=data['Vertica'],
        textposition='outside',
        name='Vertica',
        marker_color= "blue",
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