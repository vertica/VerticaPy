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

Python Statsmodels
~~~~~~~~~~~~~~~~~~~

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

Comparison
^^^^^^^^^^^

Training Run Time
------------------

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
--------------------

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
--------------------

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


