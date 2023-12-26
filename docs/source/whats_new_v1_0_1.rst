.. _whats_new_v1_0_1:

===============
Version 1.0.1
===============

This release marks the first minor update of Version 1.0.0. It encompasses all the features introduced in 1.0.0 (see :ref:`whats_new_v1_0_0`) and introduces additional functionalities along with more precise docstrings.

Options
--------

:py:func:`~verticapy.set_option` function has more options:
  - ``max_cellwidth``: Maximum width of any VerticaPy table's cell.
  - ``max_tableheight``: Maximum height of VerticaPy tables.
  - ``theme``: Theme used to display the VerticaPy objects ('light' or 'dark').

QueryProfiler
--------------

:py:class:`~verticapy.performance.vertica.qprof.QueryProfiler` class offers an extended set of functionalities, enabling the creation of complex trees with multiple metrics.
  
Others
-------

- Docstrings have been enriched to add examples and other details that will help in creating a more helpful doc.
  
Internal
=========

- Hints have been added to most functions to make sure the correct inputs are passed to all the functions.

- Updated the workflow to use the latest version of GitHub actions, and added a tox.ini file and the contributing folder.