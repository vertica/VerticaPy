
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

Vertica vs Spark
~~~~~~~~~~~~~~~~~

.. important::

  |  *Version Details*
  |  **Vertica:** 8.0.1
  |  **Spark:** 2.02

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

    Vertica: ``max iter = 100``, ``e = 10^-6``

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

    Spark: ``max iter = 100``, ``e = 10^-6``




Comparison
^^^^^^^^^^^

.. csv-table::
  :file: /_static/benchmark_lr_table.csv
  :header-rows: 2

Browse through the tabs to see the time comparison:


.. tab:: BFGS

    .. tab:: 1B
        
        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [1748.51, 2568.68]
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
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_bfgs_1b.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_bfgs_1b.html

    .. tab:: 100M

        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [84.7, 216]
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
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_bfgs_100m.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_bfgs_100m.html
    
    .. tab:: 10M

        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [26.39, 96.98]
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
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_bfgs_10m.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_bfgs_10m.html

    .. tab:: 1M

        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [4.49, 1.43]
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
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_bfgs_1m.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_bfgs_1m.html


.. tab:: Newton

    .. tab:: 1B

        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [1808.56, 1788.75]
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
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_newton_1b.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_newton_1b.html

    .. tab:: 100M

        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [85.93, 68.47]
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
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_newton_100m.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_newton_100m.html
    
    .. tab:: 10M

        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [82.60, 2.09]
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
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_newton_10m.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_newton_10m.html

    .. tab:: 1M

        .. ipython:: python
            :suppress:

            import plotly.graph_objects as go
            labels = ['Vertica', 'Spark']
            heights = [4.81, 0.7]
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
            fig.write_html("SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_newton_1m.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/benchmark_linear_regression_spark_newton_1m.html
