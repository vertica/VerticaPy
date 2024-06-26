.. _whats_new_v1_0_4:

===============
Version 1.0.4
===============

This minor release has some significant feature additions with other changes. Some salient ones are listed below:

QueryProfiler
--------------


- Now there is a new parameter ``use_temp_relation`` which allows users to display the temporary relations separately in the ``get_qplan_tree``.
- Query start and stop time are also included in the ``get_queries`` table.
- New queries can be added to the same schema and key using the ``insert`` function. 


.. code-block::

  qprof = QueryProfiler((109090909, 1))
  qprof.insert(transactions = (41823718, 2))


QueryProfilerInterface
-----------------------

- Made the metric selecting widgets in the QueryProfilerInterface to be more intuitive.
- A new widget allows to directly jump to the specific query without having to press ``Next`` button multiple times.
- A success flag is added to the display to confirm if the query was run successfully.
- Added unit for the query execution time.


Machine Learning
-----------------

- Added Vector Auto Regression (VAR) to the list of Vertica algorithms.



vDataFrame
------------

- The ``Up`` and ``Down`` arrow of the ``vDataFrame.idisplay()`` are now larger in size. 


Bugfixes
---------

- Bugfix for auto-token refresh using OAuth.
- Empty vDataFrame can now be created.

.. code-block::

    import pandas as pd
    from verticapy import read_pandas

    df = pd.DataFrame()
    df = df.reindex(columns = ["col1", "col2"])
    read_pandas(df)

- Some unit tests were modified to make them consistent with the test environment.
