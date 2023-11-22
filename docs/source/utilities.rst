.. _api.utilities:

============
Utilities
============

______

Performance
-------------

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
   QueryProfiler.get_qplan_profile
   QueryProfiler.get_qsteps
   QueryProfiler.get_request
   QueryProfiler.get_rp_status
   QueryProfiler.get_version
   QueryProfiler.step


Options & Displaying
--------------------------

.. currentmodule:: verticapy

.. autosummary:: 
   :toctree: api/

   set_option



______

Help 
-------

.. currentmodule:: verticapy

.. autosummary:: 
   :toctree: api/

   help_start
   .. version