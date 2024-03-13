.. _api.performance.vertica:

=========
Vertica
=========


Query Performance Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: verticapy.performance.vertica.qprof_interface

.. autosummary:: 
   :toctree: api/

   QueryProfilerInterface

.. currentmodule:: verticapy.performance.vertica.qprof_interface

**Methods:**

.. autosummary::
   :toctree: api/

   QueryProfilerInterface.get_cluster_config
   QueryProfilerInterface.get_cpu_time
   QueryProfilerInterface.get_qduration
   QueryProfilerInterface.get_qexecution
   QueryProfilerInterface.get_qexecution_report
   QueryProfilerInterface.get_qplan
   QueryProfilerInterface.get_qplan_tree
   QueryProfilerInterface.get_qplan_profile
   QueryProfilerInterface.get_qsteps
   QueryProfilerInterface.get_request
   QueryProfilerInterface.get_query_events
   QueryProfilerInterface.get_rp_status
   QueryProfilerInterface.get_table
   QueryProfilerInterface.get_version
   QueryProfilerInterface.next
   QueryProfilerInterface.previous
   QueryProfilerInterface.step

Query Performance
^^^^^^^^^^^^^^^^^^

.. currentmodule:: verticapy.performance.vertica.qprof

.. autosummary:: 
   :toctree: api/

   QueryProfiler

.. currentmodule:: verticapy.performance.vertica.qprof

**Methods:**

.. autosummary::
   :toctree: api/

   QueryProfiler.get_cluster_config
   QueryProfiler.get_cpu_time
   QueryProfiler.get_qduration
   QueryProfiler.get_qexecution
   QueryProfiler.get_qexecution_report
   QueryProfiler.get_qplan
   QueryProfiler.get_qplan_tree
   QueryProfiler.get_qplan_profile
   QueryProfiler.get_qsteps
   QueryProfiler.get_request
   QueryProfiler.get_query_events
   QueryProfiler.get_rp_status
   QueryProfiler.get_table
   QueryProfiler.get_version
   QueryProfiler.next
   QueryProfiler.previous
   QueryProfiler.step