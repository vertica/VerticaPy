.. _whats_new_v1_1_0:

===============
Version 1.0.1
===============

This release encompasses all the features introduced in 1.0.0 (see :ref:`whats_new_v1_0_0`) 
and introduces additional functionalities along with more precise docstrings.

Options
--------

:py:func:`~verticapy.set_option` function has more options:
  - ``max_cellwidth``: Maximum width of any VerticaPy table's cell.
  - ``max_tableheight``: Maximum height of VerticaPy tables.
  - ``theme``: Theme used to display the VerticaPy objects ('light', 'dark' or 'sphinx').

QueryProfilerInterface
-----------------------

Now we have added the functionality of a ``session_control_param`` parameter.
This allows users to enter the alter session SQL before profiling the queries.


QueryProfilerInterface
-----------------------

 - Added many more metrics for the profiled queries such as:

  - rows_filtered_sip
  - container_rows_filtered_sip
  - container_rows_pruned_sip
  
  and more...

- There is a new tab which helps you select particular tooltips from any select path id.
- A new tab also highlights if there are any non-default ``SESSION PARAMETERS``.
- Improved the efficiency of plotting the tree by chaching results.

QueryProfilerComparison
-----------------------

 - :py:class:`~verticapy.performance.vertica.qprof.QueryProfilerComparison` class offers an extended set of functionalities, enabling the creation of complex trees with multiple metrics.
  
.. code-block:: python
    
  from verticapy.performance.vertica import QueryProfilerInterface

  qprof_interface_1 = QueryProfilerInterface(
      key_id='key_1',
      target_schema='schema_1',
      )

  qprof_interface_2 = QueryProfilerInterface(
      key_id='key_2',
      target_schema='schema_1',
      )

  from verticapy.performance.vertica import QueryProfilerComparison

  qprof_compare = QueryProfilerComparison(qprof_interface_1,qprof_interface_2)

  qprof_compare.get_qplan_tree()


Others
-------

- Docstrings have been enriched to add examples and other details that will help in creating a more helpful doc.
  
Internal
=========

- Hints have been added to most functions to make sure the correct inputs are passed to all the functions.

- Updated the workflow to use the latest version of GitHub actions, and added a tox.ini file and the contributing folder.

- Some old unit tests have been deleted which are covered in the new unit tests inside the ``tests_new`` folder.