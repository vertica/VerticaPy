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
        - vCPU(per node)
        - Memory(per node)
        - Deploy Mode
        - OS
        - OS Version
        - Processor freq. (per node)
        - Processor cores (per node) 
      * - 23.3.0-5
        - On-Premises VM
        - 4 node 
        - N/A
        - 755 GB
        - Enterprise
        - Red Hat Enterprise Linux  
        - 8.7 (Ootpa)   
        - 2.3 GHz  
        - 36, 2 threads per core

.. tab:: Python

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
      * - 0.14.0
        - N/A
        - N/A
        - N/A
        - 755 GB
        - N/A
        - Red Hat Enterprise Linux  
        - 8.7 (Ootpa)   
        - 2.3 GHz  
        - N/A


**Parameters:** p = 5, d = 1, q = 1

.. csv-table:: Table Title
  :file: /_static/benchmark_arima.csv
  :header-rows: 3

.. note::

  MSE (Mean Squared Error) for Vertica is from summary table (``GET_MODEL_SUMMARY``).

Comparison
^^^^^^^^^^^

Browse throught the different tabs to see the results:

.. tab:: Up to 1M

  .. tab:: Training Run Time

    .. ipython:: python
      :suppress:

      import plotly.express as px
      import pandas as pd
      df = pd.DataFrame({
          "Size": ["10K", "100K", "1M"],
          "Vertica": [0.022, 0.055, 0.515],
          "Python": [0.064, 0.745, 8.923]
      })
      fig = px.bar(df, x="Size", y=["Vertica", "Python"], title="Vertica vs Python Performance",
        labels={"value": "Time (minutes)", "variable": "Environment", "Size": "Data Size"},
        barmode="group",
        color_discrete_map={"Vertica": #1A6AFF, "Python": #f0d917},
      )
      fig.update_layout(width = 550)
      fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_arima_train_1m.html")

    .. raw:: html
      :file: /project/data/VerticaPy/docs/figures/benchmark_arima_train_1m.html

  .. tab:: Prediction Run Time

    .. ipython:: python
      :suppress:

      import plotly.express as px
      import pandas as pd
      df = pd.DataFrame({
          "Size": ["10K", "100K", "1M"],
          "Vertica": [0.028, 0.056, 0.364],
          "Python": [0.006, 0.019, 0.027]
      })

      fig = px.bar(df, x="Size", y=["Vertica", "Python"], title="Vertica vs Python Performance",
        labels={"value": "Time (minutes)", "variable": "Environment", "Size": "Data Size"},
        barmode="group",
        color_discrete_map={"Vertica": #1A6AFF, "Python": #f0d917},
      )
      fig.update_layout(width = 550)
      fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_arima_prediction_1m.html")

    .. raw:: html
      :file: /project/data/VerticaPy/docs/figures/benchmark_arima_prediction_1m.html


  .. tab:: Mean Squared Error

    .. ipython:: python
      :suppress:

      import plotly.express as px
      import pandas as pd
      df = pd.DataFrame({
          "Size": ["10K", "100K", "1M"],
          "Vertica": [24.54, 30.53, 27.94],
          "Python": [24.6, 24.97, 25]
      })
      fig = px.bar(df, x="Size", y=["Vertica", "Python"], title="Vertica vs Python Performance",
        labels={"value": "Time (minutes)", "variable": "Environment", "Size": "Data Size"},
        barmode="group",
        color_discrete_map={"Vertica": #1A6AFF, "Python": #f0d917},
      )
      fig.update_layout(width = 550)
      fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_arima_mse_1m.html")

    .. raw:: html
      :file: /project/data/VerticaPy/docs/figures/benchmark_arima_mse_1m.html



.. tab:: More than 1M

  .. tab:: Training Run Time

    .. ipython:: python
      :suppress:

      import plotly.express as px
      import pandas as pd
      df = pd.DataFrame({
          "Size": ["10M", "100M"],
          "Vertica": [4.775, 157.763],
          "Python": [93.307, 1123.966]
      })
      fig = px.bar(df, x="Size", y=["Vertica", "Python"], title="Vertica vs Python Performance",
        labels={"value": "Time (minutes)", "variable": "Environment", "Size": "Data Size"},
        barmode="group",
        color_discrete_map={"Vertica": #1A6AFF, "Python": #f0d917},
      )
      fig.update_layout(width = 550)
      fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_arima_train_100m.html")

    .. raw:: html
      :file: /project/data/VerticaPy/docs/figures/benchmark_arima_train_100m.html

  .. tab:: Prediction Run Time

    .. ipython:: python
      :suppress:

      import plotly.express as px
      import pandas as pd
      df = pd.DataFrame({
          "Size": ["10M", "100M"],
          "Vertica": [3.785, 57.052],
          "Python": [0.333, 5.422]
      })

      fig = px.bar(df, x="Size", y=["Vertica", "Python"], title="Vertica vs Python Performance",
        labels={"value": "Time (minutes)", "variable": "Environment", "Size": "Data Size"},
        barmode="group",
        color_discrete_map={"Vertica": #1A6AFF, "Python": #f0d917},
      )
      fig.update_layout(width = 550)
      fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_arima_prediction_100m.html")

    .. raw:: html
      :file: /project/data/VerticaPy/docs/figures/benchmark_arima_prediction_100m.html


  .. tab:: Mean Squared Error

    .. ipython:: python
      :suppress:

      import plotly.express as px
      import pandas as pd
      df = pd.DataFrame({
          "Size": ["10M", "100M"],
          "Vertica": [28.52, 32.66],
          "Python": [24.99, 24.99]
      })
      fig = px.bar(df, x="Size", y=["Vertica", "Python"], title="Vertica vs Python Performance",
        labels={"value": "Time (minutes)", "variable": "Environment", "Size": "Data Size"},
        barmode="group",
        color_discrete_map={"Vertica": #1A6AFF, "Python": #f0d917},
      )
      fig.update_layout(width = 550)
      fig.write_html("/project/data/VerticaPy/docs/figures/benchmark_arima_mse_100m.html")

    .. raw:: html
      :file: /project/data/VerticaPy/docs/figures/benchmark_arima_mse_100m.html


