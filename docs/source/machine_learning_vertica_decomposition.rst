.. _api.machine_learning.vertica.decomposition:

===================================
Decomposition & Preprocessing
===================================


______

Decomposition
--------------

PCA
~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   decomposition.PCA

.. currentmodule:: verticapy.machine_learning.vertica.decomposition

**Methods:**

.. autosummary::
   :toctree: api/

   PCA.contour
   PCA.deployInverseSQL
   PCA.deploySQL
   PCA.does_model_exists
   PCA.drop
   PCA.export_models
   PCA.fit
   PCA.get_attributes
   PCA.get_match_index
   PCA.get_params
   PCA.get_plotting_lib
   PCA.get_vertica_attributes
   PCA.import_models
   PCA.inverse_transform
   PCA.plot
   PCA.plot_circle
   PCA.plot_scree
   PCA.register
   PCA.score
   PCA.set_params
   PCA.summarize
   PCA.to_binary
   PCA.to_memmodel
   PCA.to_pmml
   PCA.to_python
   PCA.to_sql
   PCA.to_tf
   PCA.transform

**Attributes:**

.. autosummary::
   :toctree: api/

   PCA.object_type

SVD
~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   decomposition.SVD

.. currentmodule:: verticapy.machine_learning.vertica.decomposition

**Methods:**

.. autosummary::
   :toctree: api/

   SVD.contour
   SVD.deployInverseSQL
   SVD.deploySQL
   SVD.does_model_exists
   SVD.drop
   SVD.export_models
   SVD.fit
   SVD.get_attributes
   SVD.get_match_index
   SVD.get_params
   SVD.get_plotting_lib
   SVD.get_vertica_attributes
   SVD.import_models
   SVD.inverse_transform
   SVD.plot
   SVD.plot_circle
   SVD.plot_scree
   SVD.register
   SVD.score
   SVD.set_params
   SVD.summarize
   SVD.to_binary
   SVD.to_memmodel
   SVD.to_pmml
   SVD.to_python
   SVD.to_sql
   SVD.to_tf
   SVD.transform

**Attributes:**

.. autosummary::
   :toctree: api/

   SVD.object_type

MCA (Beta)
~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   decomposition.MCA

.. currentmodule:: verticapy.machine_learning.vertica.decomposition

**Methods:**

.. autosummary::
   :toctree: api/

   MCA.contour
   MCA.deployInverseSQL
   MCA.deploySQL
   MCA.does_model_exists
   MCA.drop
   MCA.export_models
   MCA.fit
   MCA.get_attributes
   MCA.get_match_index
   MCA.get_params
   MCA.get_plotting_lib
   MCA.get_vertica_attributes
   MCA.import_models
   MCA.inverse_transform
   MCA.plot
   MCA.plot_circle
   MCA.plot_contrib
   MCA.plot_cos2
   MCA.plot_scree
   MCA.plot_var
   MCA.register
   MCA.score
   MCA.set_params
   MCA.summarize
   MCA.to_binary
   MCA.to_memmodel
   MCA.to_pmml
   MCA.to_python
   MCA.to_sql
   MCA.to_tf
   MCA.transform

**Attributes:**

.. autosummary::
   :toctree: api/

   MCA.object_type
____

Preprocessing 
---------------


One-Hot Encoder
~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   preprocessing.OneHotEncoder

.. currentmodule:: verticapy.machine_learning.vertica.preprocessing

**Methods:**

.. autosummary::
   :toctree: api/

   OneHotEncoder.deployInverseSQL
   OneHotEncoder.deploySQL
   OneHotEncoder.does_model_exists
   OneHotEncoder.drop
   OneHotEncoder.export_models
   OneHotEncoder.fit
   OneHotEncoder.get_attributes
   OneHotEncoder.get_match_index
   OneHotEncoder.get_params
   OneHotEncoder.get_plotting_lib
   OneHotEncoder.get_vertica_attributes
   OneHotEncoder.import_models
   OneHotEncoder.inverse_transform
   OneHotEncoder.register
   OneHotEncoder.set_params
   OneHotEncoder.summarize
   OneHotEncoder.to_binary
   OneHotEncoder.to_memmodel
   OneHotEncoder.to_pmml
   OneHotEncoder.to_python
   OneHotEncoder.to_sql
   OneHotEncoder.to_tf
   OneHotEncoder.transform


**Attributes:**

.. autosummary::
   :toctree: api/

   OneHotEncoder.object_type

______


Scaler
~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   preprocessing.Scaler

.. currentmodule:: verticapy.machine_learning.vertica.preprocessing

**Methods:**

.. autosummary::
   :toctree: api/

   Scaler.deployInverseSQL
   Scaler.deploySQL
   Scaler.does_model_exists
   Scaler.drop
   Scaler.export_models
   Scaler.fit
   Scaler.get_attributes
   Scaler.get_match_index
   Scaler.get_params
   Scaler.get_plotting_lib
   Scaler.get_vertica_attributes
   Scaler.import_models
   Scaler.inverse_transform
   Scaler.register
   Scaler.set_params
   Scaler.summarize
   Scaler.to_binary
   Scaler.to_memmodel
   Scaler.to_pmml
   Scaler.to_python
   Scaler.to_sql
   Scaler.to_tf
   Scaler.transform

**Attributes:**

.. autosummary::
   :toctree: api/

   Scaler.object_type


Standard Scaler
~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   preprocessing.StandardScaler

.. currentmodule:: verticapy.machine_learning.vertica.preprocessing

**Methods:**

.. autosummary::
   :toctree: api/

   StandardScaler.deployInverseSQL
   StandardScaler.deploySQL
   StandardScaler.does_model_exists
   StandardScaler.drop
   StandardScaler.export_models
   StandardScaler.fit
   StandardScaler.get_attributes
   StandardScaler.get_match_index
   StandardScaler.get_params
   StandardScaler.get_plotting_lib
   StandardScaler.get_vertica_attributes
   StandardScaler.import_models
   StandardScaler.inverse_transform
   StandardScaler.register
   StandardScaler.set_params
   StandardScaler.summarize
   StandardScaler.to_binary
   StandardScaler.to_memmodel
   StandardScaler.to_pmml
   StandardScaler.to_python
   StandardScaler.to_sql
   StandardScaler.to_tf
   StandardScaler.transform

**Attributes:**

.. autosummary::
   :toctree: api/

   StandardScaler.object_type


Min Max Scaler
~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   preprocessing.MinMaxScaler

.. currentmodule:: verticapy.machine_learning.vertica.preprocessing

**Methods:**

.. autosummary::
   :toctree: api/

   MinMaxScaler.contour
   MinMaxScaler.deployInverseSQL
   MinMaxScaler.deploySQL
   MinMaxScaler.does_model_exists
   MinMaxScaler.drop
   MinMaxScaler.export_models
   MinMaxScaler.fit
   MinMaxScaler.get_attributes
   MinMaxScaler.get_match_index
   MinMaxScaler.get_params
   MinMaxScaler.get_plotting_lib
   MinMaxScaler.get_vertica_attributes
   MinMaxScaler.import_models
   MinMaxScaler.inverse_transform
   MinMaxScaler.register
   MinMaxScaler.set_params
   MinMaxScaler.summarize
   MinMaxScaler.to_binary
   MinMaxScaler.to_memmodel
   MinMaxScaler.to_pmml
   MinMaxScaler.to_python
   MinMaxScaler.to_sql
   MinMaxScaler.to_tf
   MinMaxScaler.transform

**Attributes:**

.. autosummary::
   :toctree: api/

   MinMaxScaler.object_type

Robust Scaler
~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   preprocessing.RobustScaler

.. currentmodule:: verticapy.machine_learning.vertica.preprocessing

**Methods:**

.. autosummary::
   :toctree: api/

   RobustScaler.contour
   RobustScaler.deployInverseSQL
   RobustScaler.deploySQL
   RobustScaler.does_model_exists
   RobustScaler.drop
   RobustScaler.export_models
   RobustScaler.fit
   RobustScaler.get_attributes
   RobustScaler.get_match_index
   RobustScaler.get_params
   RobustScaler.get_plotting_lib
   RobustScaler.get_vertica_attributes
   RobustScaler.import_models
   RobustScaler.inverse_transform
   RobustScaler.register
   RobustScaler.set_params
   RobustScaler.summarize
   RobustScaler.to_binary
   RobustScaler.to_memmodel
   RobustScaler.to_pmml
   RobustScaler.to_python
   RobustScaler.to_sql
   RobustScaler.to_tf
   RobustScaler.transform

**Attributes:**

.. autosummary::
   :toctree: api/

   RobustScaler.object_type

_____

Balance
~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   preprocessing.balance

_____


Density Estimation
------------------

Kernel Density (Beta)
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   neighbors.KernelDensity

.. currentmodule:: verticapy.machine_learning.vertica.neighbors

**Methods:**

.. autosummary::
   :toctree: api/

   KernelDensity.contour
   KernelDensity.deploySQL
   KernelDensity.does_model_exists
   KernelDensity.drop
   KernelDensity.export_models
   KernelDensity.features_importance
   KernelDensity.fit
   KernelDensity.get_attributes
   KernelDensity.get_match_index
   KernelDensity.get_params
   KernelDensity.get_plotting_lib
   KernelDensity.get_score
   KernelDensity.get_tree
   KernelDensity.get_vertica_attributes
   KernelDensity.import_models
   KernelDensity.plot
   KernelDensity.plot_tree
   KernelDensity.predict
   KernelDensity.register
   KernelDensity.regression_report
   KernelDensity.report
   KernelDensity.score
   KernelDensity.set_params
   KernelDensity.summarize
   KernelDensity.to_binary
   KernelDensity.to_graphviz
   KernelDensity.to_pmml
   KernelDensity.to_python
   KernelDensity.to_sql
   KernelDensity.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   KernelDensity.object_type
______