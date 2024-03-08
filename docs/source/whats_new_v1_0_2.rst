.. _whats_new_v1_0_2:

===============
Version 1.0.2
===============

This minor release has some significant feature additions with other changes. Some salient ones are listed below:

Pipelines (Beta)
-----------------

VerticaPy now has **Pipelines**! 

- ``Pipelines`` is a YAML-based configuration for defining machine learning workflows, simplifying the process of setting up and managing machine learning pipelines.
- For beginners, it provides an easy-to-learn alternative to Python and SQL reducing the initial barriers to entry for creating models.
- For more experienced users, it offers templating features to enhance modularity, minimize errors, and promote efficient code reuse in machine learning projects.


Performance
------------

- We have enhanced the QueryProfiler to improve its robustness. :py:func:`~verticapy.performance.vertica.QueryProfiler`.
- Introducing a completely new **Query Profiler Interface**, enabling users to navigate through various queries and access them without the need to re-enter all the code. All of this can be accomplished using only your mouse within Jupyter Notebook environments. For more information please look at :py:func:`~verticapy.performance.vertica.QueryProfilerInterface`.

These updates significantly enhance the accessibility, debugging, and enhancement capabilities of your queries.

OAuth Refresh Tokens
---------------------

- We have updated the connector to accept OAuth refresh tokens.
- Additioanlly we have added a ``prompt`` option for :py:func:`~verticapy.connection.new_connection`. This allows the user to enter the secrets discretly with a masked display.

Multi-TimeSeries (Beta)
-----------------------

We added a new Time Series class: ``TimeSeriesByCategory``. This allows the users to build multiple models based off on a category. The number of models created
are equal to the categories. This saves users time to create multiple models separately. For more inofrmation please see :py:func:`~verticapy.machine_learning.vertica.tsa.ensemble.TimeSeriesByCategory`.

Plots
------

- Two new plots have been added for plotly that were previously missing:

  - :py:func:`~verticapy.machine_learning.vertica.decomposition.plot_scree`
  - :py:func:`~verticapy.machine_learning.vertica.decomposition.plot_var`
  
Unit Tests
-----------

- We continue to shift our old tests to the new more robust format. 

Examples
---------

- Most of the `examples <https://github.com/vertica/VerticaPy/tree/master/examples>`_ have been updated with the latest verticapy format. 
