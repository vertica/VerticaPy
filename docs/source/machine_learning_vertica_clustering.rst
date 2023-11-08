.. _api.machine_learning.vertica.clustering:

===============================
Clustering & Anomaly Detection
===============================

______

Clustering
-----------


K-Means
~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   cluster.KMeans

.. currentmodule:: verticapy.machine_learning.vertica.cluster

**Methods:**

.. autosummary::
   :toctree: api/

   KMeans.contour
   KMeans.deploySQL
   KMeans.does_model_exists
   KMeans.drop
   KMeans.fit
   KMeans.get_attributes
   KMeans.get_match_index
   KMeans.get_params
   KMeans.get_plotting_lib
   KMeans.get_vertica_attributes
   KMeans.plot
   KMeans.plot_voronoi
   KMeans.predict
   KMeans.register
   KMeans.set_params
   KMeans.summarize
   KMeans.to_memmodel
   KMeans.to_python
   KMeans.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   KMeans.object_type

K-Prototype
~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   cluster.KPrototypes

.. currentmodule:: verticapy.machine_learning.vertica.cluster

**Methods:**

.. autosummary::
   :toctree: api/

   KPrototypes.to_memmodel

Bisecting K-Means
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   cluster.BisectingKMeans

.. currentmodule:: verticapy.machine_learning.vertica.cluster

**Methods:**

.. autosummary::
   :toctree: api/

   BisectingKMeans.contour
   BisectingKMeans.deploySQL
   BisectingKMeans.does_model_exists
   BisectingKMeans.drop
   BisectingKMeans.features_importance
   BisectingKMeans.fit
   BisectingKMeans.get_attributes
   BisectingKMeans.get_match_index
   BisectingKMeans.get_params
   BisectingKMeans.get_plotting_lib
   BisectingKMeans.get_vertica_attributes
   BisectingKMeans.plot
   BisectingKMeans.plot_tree
   BisectingKMeans.plot_voronoi
   BisectingKMeans.predict
   BisectingKMeans.register
   BisectingKMeans.set_params
   BisectingKMeans.summarize  
   BisectingKMeans.to_memmodel
   BisectingKMeans.to_python
   BisectingKMeans.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   BisectingKMeans.object_type


DBSCAN (Beta)
~~~~~~~~~~~~~~


.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   cluster.DBSCAN

.. currentmodule:: verticapy.machine_learning.vertica.cluster

**Methods:**

.. autosummary::
   :toctree: api/

   DBSCAN.contour
   DBSCAN.deploySQL
   DBSCAN.does_model_exists
   DBSCAN.drop
   DBSCAN.fit
   DBSCAN.get_attributes
   DBSCAN.get_match_index
   DBSCAN.get_params
   DBSCAN.get_plotting_lib
   DBSCAN.get_vertica_attributes
   DBSCAN.plot
   DBSCAN.predict
   DBSCAN.register
   DBSCAN.set_params
   DBSCAN.summarize
   DBSCAN.to_python
   DBSCAN.to_sql

_____________

Anomaly Detection
-------------------

Isolation Forest
~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   ensemble.IsolationForest

.. currentmodule:: verticapy.machine_learning.vertica.ensemble

**Methods:**

.. autosummary::
   :toctree: api/

   IsolationForest.contour
   IsolationForest.deploySQL
   IsolationForest.does_model_exists
   IsolationForest.drop
   IsolationForest.features_importance
   IsolationForest.fit
   IsolationForest.get_attributes
   IsolationForest.get_match_index
   IsolationForest.get_params
   IsolationForest.get_plotting_lib
   IsolationForest.get_vertica_attributes
   IsolationForest.plot
   IsolationForest.predict
   IsolationForest.register
   IsolationForest.set_params
   IsolationForest.summarize
   IsolationForest.to_memmodel
   IsolationForest.to_python
   IsolationForest.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   IsolationForest.object_type

Local Outlier Factor (Beta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   neighbors.LocalOutlierFactor

.. currentmodule:: verticapy.machine_learning.vertica.neighbors.LocalOutlierFactor

**Methods:**

.. autosummary::
   :toctree: api/

   drop
   fit
   plot
   predict
   


