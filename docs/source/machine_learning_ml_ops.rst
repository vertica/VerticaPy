.. _api.machine_learning.ml_ops:

=========
ML Ops
=========



______



Model Versioning
-----------------


.. currentmodule:: verticapy.mlops
   
.. autosummary::
   :toctree: api/

   model_versioning.RegisteredModel

.. currentmodule:: verticapy.mlops.model_versioning

**Methods:**

.. autosummary::
   :toctree: api/

   RegisteredModel.change_status
   RegisteredModel.list_models
   RegisteredModel.list_status_history
   RegisteredModel.predict
   RegisteredModel.predict_proba


______



Model Tracking
-----------------

.. autoclass:: verticapy.mlops.
    :members:

.. currentmodule:: verticapy.mlops
   
.. autosummary::
   :toctree: api/

   model_tracking.vExperiment

.. currentmodule:: verticapy.mlops.model_tracking

**Methods:**

.. autosummary::
   :toctree: api/

   vExperiment.add_model
   vExperiment.drop
   vExperiment.get_plotting_lib
   vExperiment.list_models
   vExperiment.load_best_model
   vExperiment.plot


    