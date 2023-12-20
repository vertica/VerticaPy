.. _api.machine_learning.vertica.clustering:

===============================
Clustering & Anomaly Detection
===============================

______

Clustering
----------

K-Means
~~~~~~~

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
   KMeans.export_models
   KMeans.fit
   KMeans.get_attributes
   KMeans.get_match_index
   KMeans.get_params
   KMeans.get_plotting_lib
   KMeans.get_vertica_attributes
   KMeans.import_models
   KMeans.plot
   KMeans.plot_voronoi
   KMeans.predict
   KMeans.register
   KMeans.set_params
   KMeans.summarize
   KMeans.to_binary
   KMeans.to_memmodel
   KMeans.to_pmml
   KMeans.to_python
   KMeans.to_sql
   KMeans.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   KMeans.object_type

K-Prototype
~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   cluster.KPrototypes

.. currentmodule:: verticapy.machine_learning.vertica.cluster

**Methods:**

.. autosummary::
   :toctree: api/

   KPrototypes.contour
   KPrototypes.deploySQL
   KPrototypes.does_model_exists
   KPrototypes.drop
   KPrototypes.export_models
   KPrototypes.fit
   KPrototypes.get_attributes
   KPrototypes.get_match_index
   KPrototypes.get_params
   KPrototypes.get_plotting_lib
   KPrototypes.get_vertica_attributes
   KPrototypes.import_models
   KPrototypes.plot
   KPrototypes.plot_voronoi
   KPrototypes.predict
   KPrototypes.register
   KPrototypes.set_params
   KPrototypes.summarize
   KPrototypes.to_binary
   KPrototypes.to_memmodel
   KPrototypes.to_pmml
   KPrototypes.to_python
   KPrototypes.to_sql
   KPrototypes.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   KPrototypes.object_type

Bisecting K-Means
~~~~~~~~~~~~~~~~~

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
   BisectingKMeans.export_models
   BisectingKMeans.features_importance
   BisectingKMeans.fit
   BisectingKMeans.get_attributes
   BisectingKMeans.get_match_index
   BisectingKMeans.get_params
   BisectingKMeans.get_plotting_lib
   BisectingKMeans.get_score
   BisectingKMeans.get_tree
   BisectingKMeans.get_vertica_attributes
   BisectingKMeans.import_models
   BisectingKMeans.plot
   BisectingKMeans.plot_tree
   BisectingKMeans.plot_voronoi
   BisectingKMeans.predict
   BisectingKMeans.register
   BisectingKMeans.set_params
   BisectingKMeans.summarize
   BisectingKMeans.to_binary
   BisectingKMeans.to_graphviz
   BisectingKMeans.to_memmodel
   BisectingKMeans.to_pmml
   BisectingKMeans.to_python
   BisectingKMeans.to_sql
   BisectingKMeans.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   BisectingKMeans.object_type

DBSCAN (Beta)
~~~~~~~~~~~~~

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
   DBSCAN.export_models
   DBSCAN.fit
   DBSCAN.get_attributes
   DBSCAN.get_match_index
   DBSCAN.get_params
   DBSCAN.get_plotting_lib
   DBSCAN.get_vertica_attributes
   DBSCAN.import_models
   DBSCAN.plot
   DBSCAN.predict
   DBSCAN.register
   DBSCAN.set_params
   DBSCAN.summarize
   DBSCAN.to_binary
   DBSCAN.to_pmml
   DBSCAN.to_python
   DBSCAN.to_sql
   DBSCAN.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   DBSCAN.object_type

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
   IsolationForest.decision_function
   IsolationForest.deploySQL
   IsolationForest.does_model_exists
   IsolationForest.drop
   IsolationForest.export_models
   IsolationForest.features_importance
   IsolationForest.fit
   IsolationForest.get_attributes
   IsolationForest.get_match_index
   IsolationForest.get_params
   IsolationForest.get_plotting_lib
   IsolationForest.get_score
   IsolationForest.get_tree
   IsolationForest.get_vertica_attributes
   IsolationForest.import_models
   IsolationForest.plot
   IsolationForest.plot_tree
   IsolationForest.predict
   IsolationForest.register
   IsolationForest.set_params
   IsolationForest.summarize
   IsolationForest.to_binary
   IsolationForest.to_graphviz
   IsolationForest.to_memmodel
   IsolationForest.to_pmml
   IsolationForest.to_python
   IsolationForest.to_sql
   IsolationForest.to_tf

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

.. currentmodule:: verticapy.machine_learning.vertica.neighbors

**Methods:**

.. autosummary::
   :toctree: api/

   LocalOutlierFactor.contour
   LocalOutlierFactor.deploySQL
   LocalOutlierFactor.does_model_exists
   LocalOutlierFactor.drop
   LocalOutlierFactor.export_models
   LocalOutlierFactor.fit
   LocalOutlierFactor.get_attributes
   LocalOutlierFactor.get_match_index
   LocalOutlierFactor.get_params
   LocalOutlierFactor.get_plotting_lib
   LocalOutlierFactor.get_vertica_attributes
   LocalOutlierFactor.import_models
   LocalOutlierFactor.predict
   LocalOutlierFactor.register
   LocalOutlierFactor.set_params
   LocalOutlierFactor.summarize
   LocalOutlierFactor.to_binary
   LocalOutlierFactor.to_pmml
   LocalOutlierFactor.to_python
   LocalOutlierFactor.to_sql
   LocalOutlierFactor.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   LocalOutlierFactor.object_type