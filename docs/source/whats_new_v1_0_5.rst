.. _whats_new_v1_0_5:

===============
Version 1.0.5
===============

This minor release has some significant feature additions with other changes. Some salient ones are listed below:

QueryProfilerStats
------------------


Added a new class which calculates some of the statistics for queries that can help diagnose query performance issues.

You can call the main function to perform all the tests:

.. code-block::

  qprof = QueryProfilerStats((109090909, 1))
  qprof.main_tests()

It is also inherited by the QueryProfilerInterface class, so it can be used from that class as well. For example:

.. code-block::

  qprof = QueryProfilerInterface((109090909, 1))
  qprof.main_tests()

QueryProfiler
-----------------------

- Added the following new tables to the profile information:

  - dc_scan_events


Unit tests
-----------------

- Fixed some broken unit tests



Bugfixes
------------

- Regression metrics were corrected which were previously giving erroneous results (AIC_SCORE, BIC_SCORE, R2_SCORE).
- In certain scenarios, the vDataFrame creation was taking too much time (Issue #1235). This was resolved.
- QueryProfiler import bug fix. 

