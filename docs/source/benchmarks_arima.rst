.. _benchmarks.arima:

======
ARIMA
======

ARIMA (AutoRegressive Integrated Moving Average) models combine the 
abilities of ``AUTOREGRESSOR`` and ``MOVING_AVERAGE`` models by 
making future predictions based on both preceding time series 
values and errors of previous predictions. ARIMA models also 
provide the option to apply a differencing operation to the input 
data, which can turn a non-stationary time series into a stationary 
time series.

Vertica vs Python Statsmodels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. important::

  |  *Version Details*
  |  **Vertica:** 23.3.0-5
  |  **Python Statsmodel:** 0.14.0

The aim of this benchmark is to compare Vertica algorithm performance 
against 
`python statsmodels <https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html>`_.

Dataset
^^^^^^^

The dataset was artifically created using a Linear model with some random noise.


Test Environment
^^^^^^^^^^^^^^^^^ 

.. important::

  **Vertica Version:** 23.3.0-5

Below are the configuration on which the tests were carried out:

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
      * - 23.3.0-5
        - On-Premises VM
        - 4 node 
        - 36, 2 threads per core
        - 755 GB
        - Enterprise
        - Red Hat Enterprise Linux  
        - 8.7 (Ootpa)   
        - 2.3 GHz  

.. tab:: Python

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
      * - 0.14.0
        - N/A
        - N/A
        - N/A
        - 755 GB
        - N/A
        - Red Hat Enterprise Linux  
        - 8.7 (Ootpa)   
        - 2.3 GHz  


**Parameters:** p = 5, d = 1, q = 1

.. csv-table:: Table Title
  :file: /_static/benchmark_arima.csv
  :header-rows: 3

.. note::

  MSE (Mean Squared Error) for Vertica is from summary table (``GET_MODEL_SUMMARY``).

Comparison
^^^^^^^^^^^

Browse through the different tabs to view the results.

.. tab:: 100M

  .. tab:: Training Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [157.763, 1123.966]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_train_100m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_train_100m.html

  .. tab:: Prediction Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [57.052, 5.422]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_prediction_100m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_prediction_100m.html


  .. tab:: Mean Squared Error

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [32.66, 24.99]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='MSE'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_mse_100m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_mse_100m.html

.. tab:: 10M

  .. tab:: Training Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [4.775, 93.307]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_train_10m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_train_10m.html

  .. tab:: Prediction Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [3.785, 0.333]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_prediction_10m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_prediction_10m.html


  .. tab:: Mean Squared Error

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [28.52, 24.99]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='MSE'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_mse_10m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_mse_10m.html

.. tab:: 1M

  .. tab:: Training Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [0.515, 8.923]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_train_1m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_train_1m.html

  .. tab:: Prediction Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [0.364, 0.027]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_prediction_1m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_prediction_1m.html


  .. tab:: Mean Squared Error

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [27.94, 25]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='MSE'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_mse_1m.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_mse_1m.html

.. tab:: 100K

  .. tab:: Training Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [0.055, 0.745]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_train_100k.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_train_100k.html

  .. tab:: Prediction Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [0.056, 0.019]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_prediction_100k.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_prediction_100k.html


  .. tab:: Mean Squared Error

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [30.53, 24.97]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='MSE'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_mse_100k.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_mse_100k.html

.. tab:: 10K

  .. tab:: Training Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [0.022, 0.064]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_train_10k.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_train_10k.html

  .. tab:: Prediction Run Time

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [0.028, 0.006]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='Time (minutes)'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_prediction_10k.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_prediction_10k.html

  .. tab:: Mean Squared Error

    .. ipython:: python
      :suppress:

      import plotly.graph_objects as go
      labels = ['Vertica', 'Python']
      heights = [24.54, 24.6]
      colors = ["#1A6AFF", '#f0d917']
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
          title='Vertica vs Python',
          yaxis=dict(title='MSE'),
          width = 600,
          height = 500,
          )
      fig.write_html("SPHINX_DIRECTORY/figures/benchmark_arima_mse_10k.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/benchmark_arima_mse_10k.html