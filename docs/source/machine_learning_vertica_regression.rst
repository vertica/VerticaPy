.. _api.machine_learning.vertica.regression:

===============
Regression
===============


______


Linear Models
--------------




Linear Regression
~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   linear_model.LinearRegression

.. currentmodule:: verticapy.machine_learning.vertica

**Methods:**

.. autosummary::
   :toctree: api/

   LinearRegression.contour
   LinearRegression.deploySQL
   LinearRegression.does_model_exists
   LinearRegression.drop
   LinearRegression.features_importance
   LinearRegression.fit
   LinearRegression.get_attributes
   LinearRegression.get_match_index
   LinearRegression.get_params
   LinearRegression.get_plotting_lib
   LinearRegression.get_vertica_attributes
   LinearRegression.plot
   LinearRegression.predict
   LinearRegression.regression_report
   LinearRegression.report
   LinearRegression.score
   LinearRegression.set_params
   LinearRegression.summarize
   LinearRegression.to_memmodel
   LinearRegression.to_python
   LinearRegression.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   LinearRegression.object_type

Ridge
~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   linear_model.Ridge

.. currentmodule:: verticapy.machine_learning.vertica.linear_model

**Methods:**

.. autosummary::
   :toctree: api/

   Ridge.contour
   Ridge.deploySQL
   Ridge.does_model_exists
   Ridge.drop
   Ridge.features_importance
   Ridge.fit
   Ridge.get_attributes
   Ridge.get_match_index
   Ridge.get_params
   Ridge.get_plotting_lib
   Ridge.get_vertica_attributes
   Ridge.plot
   Ridge.predict
   Ridge.regression_report
   Ridge.report
   Ridge.score
   Ridge.set_params
   Ridge.summarize
   Ridge.to_memmodel
   Ridge.to_python
   Ridge.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   Ridge.object_type

Lasso
~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   linear_model.Lasso

.. currentmodule:: verticapy.machine_learning.vertica.linear_model

**Methods:**

.. autosummary::
   :toctree: api/

   Lasso.contour
   Lasso.deploySQL
   Lasso.does_model_exists
   Lasso.drop
   Lasso.features_importance
   Lasso.fit
   Lasso.get_attributes
   Lasso.get_match_index
   Lasso.get_params
   Lasso.get_plotting_lib
   Lasso.get_vertica_attributes
   Lasso.plot
   Lasso.predict
   Lasso.regression_report
   Lasso.report
   Lasso.score
   Lasso.set_params
   Lasso.summarize
   Lasso.to_memmodel
   Lasso.to_python
   Lasso.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   Lasso.object_type

Elastic Net
~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   linear_model.ElasticNet

.. currentmodule:: verticapy.machine_learning.vertica.linear_model

**Methods:**

.. autosummary::
   :toctree: api/

   ElasticNet.contour
   ElasticNet.deploySQL
   ElasticNet.does_model_exists
   ElasticNet.drop
   ElasticNet.features_importance
   ElasticNet.fit
   ElasticNet.get_attributes
   ElasticNet.get_match_index
   ElasticNet.get_params
   ElasticNet.get_plotting_lib
   ElasticNet.get_vertica_attributes
   ElasticNet.plot
   ElasticNet.predict
   ElasticNet.regression_report
   ElasticNet.report
   ElasticNet.score
   ElasticNet.set_params
   ElasticNet.summarize
   ElasticNet.to_memmodel
   ElasticNet.to_python
   ElasticNet.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   ElasticNet.object_type


Linear SVR
~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   svm.LinearSVR

.. currentmodule:: verticapy.machine_learning.vertica.svm

**Methods:**

.. autosummary::
   :toctree: api/

   LinearSVR.contour
   LinearSVR.deploySQL
   LinearSVR.does_model_exists
   LinearSVR.drop
   LinearSVR.features_importance
   LinearSVR.fit
   LinearSVR.get_attributes
   LinearSVR.get_match_index
   LinearSVR.get_params
   LinearSVR.get_plotting_lib
   LinearSVR.get_vertica_attributes
   LinearSVR.plot
   LinearSVR.predict
   LinearSVR.regression_report
   LinearSVR.report
   LinearSVR.score
   LinearSVR.set_params
   LinearSVR.summarize
   LinearSVR.to_memmodel
   LinearSVR.to_python
   LinearSVR.to_sql


**Attributes:**

.. autosummary::
   :toctree: api/

   LinearSVR.object_type


Poisson Regression
~~~~~~~~~~~~~~~~~~~~~~



.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   linear_model.PoissonRegressor

.. currentmodule:: verticapy.machine_learning.vertica.linear_model

**Methods:**

.. autosummary::
   :toctree: api/

   PoissonRegressor.contour
   PoissonRegressor.deploySQL
   PoissonRegressor.does_model_exists
   PoissonRegressor.drop
   PoissonRegressor.features_importance
   PoissonRegressor.fit
   PoissonRegressor.get_attributes
   PoissonRegressor.get_match_index
   PoissonRegressor.get_params
   PoissonRegressor.get_plotting_lib
   PoissonRegressor.get_vertica_attributes
   PoissonRegressor.plot
   PoissonRegressor.predict
   PoissonRegressor.regression_report
   PoissonRegressor.report
   PoissonRegressor.score
   PoissonRegressor.set_params
   PoissonRegressor.summarize
   PoissonRegressor.to_memmodel
   PoissonRegressor.to_python
   PoissonRegressor.to_sql
_____




Tree-based Models
------------------


Dummy Tree
~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   tree.DummyTreeRegressor

.. currentmodule:: verticapy.machine_learning.vertica.tree

**Methods:**

.. autosummary::
   :toctree: api/

   DummyTreeRegressor.contour
   DummyTreeRegressor.deploySQL
   DummyTreeRegressor.does_model_exists
   DummyTreeRegressor.drop
   DummyTreeRegressor.features_importance
   DummyTreeRegressor.fit
   DummyTreeRegressor.get_attributes
   DummyTreeRegressor.get_match_index
   DummyTreeRegressor.get_params
   DummyTreeRegressor.get_plotting_lib
   DummyTreeRegressor.get_score
   DummyTreeRegressor.get_tree
   DummyTreeRegressor.get_vertica_attributes
   DummyTreeRegressor.plot
   DummyTreeRegressor.plot_tree
   DummyTreeRegressor.predict
   DummyTreeRegressor.regression_report
   DummyTreeRegressor.report
   DummyTreeRegressor.score
   DummyTreeRegressor.set_params
   DummyTreeRegressor.summarize
   DummyTreeRegressor.to_memmodel
   DummyTreeRegressor.to_python
   DummyTreeRegressor.to_sql


**Attributes:**

.. autosummary::
   :toctree: api/

   DummyTreeRegressor.object_type

Decision Tree Regressor
~~~~~~~~~~~~~~~~~~~~~~~~


.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   tree.DecisionTreeRegressor

.. currentmodule:: verticapy.machine_learning.vertica.tree

**Methods:**

.. autosummary::
   :toctree: api/


   DecisionTreeRegressor.contour
   DecisionTreeRegressor.deploySQL
   DecisionTreeRegressor.does_model_exists
   DecisionTreeRegressor.drop
   DecisionTreeRegressor.features_importance
   DecisionTreeRegressor.fit
   DecisionTreeRegressor.get_attributes
   DecisionTreeRegressor.get_match_index
   DecisionTreeRegressor.get_params
   DecisionTreeRegressor.get_plotting_lib
   DecisionTreeRegressor.get_score
   DecisionTreeRegressor.get_tree
   DecisionTreeRegressor.get_vertica_attributes
   DecisionTreeRegressor.plot
   DecisionTreeRegressor.plot_tree
   DecisionTreeRegressor.predict
   DecisionTreeRegressor.regression_report
   DecisionTreeRegressor.report
   DecisionTreeRegressor.score
   DecisionTreeRegressor.set_params
   DecisionTreeRegressor.summarize
   DecisionTreeRegressor.to_memmodel
   DecisionTreeRegressor.to_python
   DecisionTreeRegressor.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   DecisionTreeRegressor.object_type


Random Forest Regressor
~~~~~~~~~~~~~~~~~~~~~~~~


.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   ensemble.RandomForestRegressor

.. currentmodule:: verticapy.machine_learning.vertica.ensemble

**Methods:**

.. autosummary::
   :toctree: api/

   RandomForestRegressor.contour
   RandomForestRegressor.deploySQL
   RandomForestRegressor.does_model_exists
   RandomForestRegressor.drop
   RandomForestRegressor.features_importance
   RandomForestRegressor.fit
   RandomForestRegressor.get_attributes
   RandomForestRegressor.get_match_index
   RandomForestRegressor.get_params
   RandomForestRegressor.get_plotting_lib
   RandomForestRegressor.get_vertica_attributes
   RandomForestRegressor.plot
   RandomForestRegressor.predict
   RandomForestRegressor.regression_report
   RandomForestRegressor.report
   RandomForestRegressor.score
   RandomForestRegressor.set_params
   RandomForestRegressor.summarize
   RandomForestRegressor.to_memmodel
   RandomForestRegressor.to_python
   RandomForestRegressor.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   RandomForestRegressor.object_type


XGB Regressor
~~~~~~~~~~~~~~~~

   
.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   ensemble.XGBRegressor

.. currentmodule:: verticapy.machine_learning.vertica.ensemble

**Methods:**

.. autosummary::
   :toctree: api/

   XGBRegressor.contour
   XGBRegressor.deploySQL
   XGBRegressor.does_model_exists
   XGBRegressor.drop
   XGBRegressor.features_importance
   XGBRegressor.fit
   XGBRegressor.get_attributes
   XGBRegressor.get_match_index
   XGBRegressor.get_params
   XGBRegressor.get_plotting_lib
   XGBRegressor.get_vertica_attributes
   XGBRegressor.plot
   XGBRegressor.predict
   XGBRegressor.regression_report
   XGBRegressor.report
   XGBRegressor.score
   XGBRegressor.set_params
   XGBRegressor.summarize
   XGBRegressor.to_memmodel
   XGBRegressor.to_python
   XGBRegressor.to_sql


**Attributes:**

.. autosummary::
   :toctree: api/

   XGBRegressor.object_type
____

Neighbors
-----------

K-Nearest Neighbors Regressor (Beta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   neighbors.KNeighborsRegressor

.. currentmodule:: verticapy.machine_learning.vertica.neighbors

**Methods:**

.. autosummary::
   :toctree: api/

   KNeighborsRegressor.contour
   KNeighborsRegressor.deploySQL
   KNeighborsRegressor.does_model_exists
   KNeighborsRegressor.drop
   KNeighborsRegressor.fit
   KNeighborsRegressor.get_attributes
   KNeighborsRegressor.get_match_index
   KNeighborsRegressor.get_params
   KNeighborsRegressor.get_plotting_lib
   KNeighborsRegressor.get_vertica_attributes
   KNeighborsRegressor.predict
   KNeighborsRegressor.regression_report
   KNeighborsRegressor.report
   KNeighborsRegressor.score
   KNeighborsRegressor.set_params
   KNeighborsRegressor.summarize
   KNeighborsRegressor.to_python
   KNeighborsRegressor.to_sql

**Attributes:**

.. autosummary::
   :toctree: api/

   KNeighborsRegressor.object_type
______

