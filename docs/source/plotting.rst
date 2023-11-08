.. _api.plotting:

=============
Plotting
=============

______


Base
------


.. currentmodule:: verticapy.plotting.base

.. autosummary:: 
   :toctree: api/

    PlottingBase

**Methods:**

.. currentmodule:: verticapy.plotting.base

.. autosummary:: 
   :toctree: api/

    PlottingBase.get_cmap
    PlottingBase.get_colors
    
______

SQL
----

.. currentmodule:: verticapy.plotting.sql

.. autosummary:: 
   :toctree: api/

    PlottingBaseSQL


_____

Switching Libaries
-------------------

Plotly
~~~~~~~~

.. code-block:: python

    verticapy.set_option("plotting_lib","plotly")

Matplotlib
~~~~~~~~~~~~

.. code-block:: python

    verticapy.set_option("plotting_lib","matplotlib")

Highcharts
~~~~~~~~~~~

.. code-block:: python

    verticapy.set_option("plotting_lib","highcharts")