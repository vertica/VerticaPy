.. _api.performance.vertica:

=========
Vertica
=========

Query Profiler
^^^^^^^^^^^^^^^

.. currentmodule:: verticapy.performance.vertica.qprof

.. autosummary:: 
   :toctree: api/

   QueryProfiler

.. currentmodule:: verticapy.performance.vertica.qprof

**Methods:**

.. autosummary::
   :toctree: api/

   QueryProfiler.export_profile
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
   QueryProfiler.get_queries
   QueryProfiler.get_query_events
   QueryProfiler.get_rp_status
   QueryProfiler.get_table
   QueryProfiler.get_version
   QueryProfiler.import_profile
   QueryProfiler.insert
   QueryProfiler.set_position
   QueryProfiler.next
   QueryProfiler.previous
   QueryProfiler.step
   QueryProfiler.to_html
   QueryProfiler.get_activity_time
   QueryProfiler.get_qplan_explain

Query Profiler Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: verticapy.performance.vertica.qprof_interface

.. autosummary:: 
   :toctree: api/

   QueryProfilerInterface

.. currentmodule:: verticapy.performance.vertica.qprof_interface

**Methods:**

.. autosummary::
   :toctree: api/

   QueryProfilerInterface.export_profile
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
   QueryProfilerInterface.get_queries
   QueryProfilerInterface.get_query_events
   QueryProfilerInterface.get_rp_status
   QueryProfilerInterface.get_table
   QueryProfilerInterface.get_version
   QueryProfilerInterface.main_tests
   QueryProfilerInterface.next
   QueryProfilerInterface.previous
   QueryProfilerInterface.set_position
   QueryProfilerInterface.step
   QueryProfilerInterface.to_html
   QueryProfilerInterface.client_data_test
   QueryProfilerInterface.clock_exec_time_test
   QueryProfilerInterface.exec_time_test
   QueryProfilerInterface.get_activity_time
   QueryProfilerInterface.get_qplan_explain
   QueryProfilerInterface.get_qsteps_
   QueryProfilerInterface.get_resource_acquisition
   QueryProfilerInterface.import_profile
   QueryProfilerInterface.pool_queue_wait_time_test
   QueryProfilerInterface.qsteps_clicked
   QueryProfilerInterface.query_events_test
   QueryProfilerInterface.segmentation_test
   QueryProfilerInterface.update_cpu_time
   QueryProfilerInterface.update_qsteps
   QueryProfilerInterface.update_step

Query Profiler Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: verticapy.performance.vertica.qprof_interface

.. autosummary:: 
   :toctree: api/

   QueryProfilerComparison

.. currentmodule:: verticapy.performance.vertica.qprof_interface

**Methods:**

.. autosummary::
   :toctree: api/

   QueryProfilerComparison.get_qplan_tree
   QueryProfilerComparison.sync_all_checkboxes
   QueryProfilerComparison.unsync_all_checkboxes