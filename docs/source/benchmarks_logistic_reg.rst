
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

Vertica vs Spark
~~~~~~~~~~~~~~~~~

.. important::

  |  *Version Details*
  |  **Vertica:** 8.0.1
  |  **Spark:** 2.02

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

.. tab:: Vertica

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
        * - 8.0.1
          - On Premise VM
          - 3 node cluster
          - 36, 2 threads per core
          - 755 GB
          - Enterprise
          - Red Hat Enterprise Linux 
          - 8.7 (Ootpa)
          - 2.4GHz

.. tab:: Spark

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
        * - 2.02
          - N/A
          - N/A
          - 36, 2 threads per core
          - 755 GB
          - N/A
          - Red Hat Enterprise Linux 
          - 8.7 (Ootpa)
          - 2.4GHz



Comparison
^^^^^^^^^^^


.. csv-table::
  :file: /_static/benchmark_logr_table.csv
  :header-rows: 2

Browse through the tabs to see the time comparison:


.. tab:: BFGS

    .. tab:: 1B
        
        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [388.89, 2222]
            colors = ["#1A6AFF", 'black']
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
                title='Data Size: 1B',
                yaxis=dict(title='Time (seconds)'),
                bargap=0.2,
                width = 600,
                height = 500
                )
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_logistic_regression_spark_bfgs_1b.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_logistic_regression_spark_bfgs_1b.html

    .. tab:: 100M

        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [36.54, 367.27]
            colors = ["#1A6AFF", 'black']
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
                yaxis=dict(title='Time (seconds)'),
                bargap=0.2,
                width = 600,
                height = 500
            )
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_logistic_regression_spark_bfgs_100m.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_logistic_regression_spark_bfgs_100m.html
    
    .. tab:: 10M

        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [45.15, 12.05]
            colors = ["#1A6AFF", 'black']
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
                title='Data Size: 10 M',
                yaxis=dict(title='Time (seconds)'),
                bargap=0.2,
                width = 600,
                height = 500
            )
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_logistic_regression_spark_bfgs_10m.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_logistic_regression_spark_bfgs_10m.html

    .. tab:: 1M

        .. ipython:: python
            :suppress:
            :okwarning:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [14.74, 4.52]
            colors = ["#1A6AFF", 'black']
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
                title='Data Size: 1M',
                yaxis=dict(title='Time (seconds)'),
                bargap=0.2,
                width = 600,
                height = 500
            )
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_logistic_regression_spark_bfgs_1m.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_logistic_regression_spark_bfgs_1m.html
