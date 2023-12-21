.. _api.machine_learning.vertica.classification:

===============
Classification
===============

______


Linear Models
-------------

Linear SVC
~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica

.. autosummary::
   :toctree: api/

   svm.LinearSVC

.. currentmodule:: verticapy.machine_learning.vertica.svm

**Methods:**

.. autosummary::
   :toctree: api/

   LinearSVC.classification_report
   LinearSVC.confusion_matrix
   LinearSVC.contour
   LinearSVC.cutoff_curve
   LinearSVC.deploySQL
   LinearSVC.does_model_exists
   LinearSVC.drop
   LinearSVC.export_models
   LinearSVC.features_importance
   LinearSVC.fit
   LinearSVC.get_attributes
   LinearSVC.get_match_index
   LinearSVC.get_params
   LinearSVC.get_plotting_lib
   LinearSVC.get_vertica_attributes
   LinearSVC.import_models
   LinearSVC.lift_chart
   LinearSVC.plot
   LinearSVC.prc_curve
   LinearSVC.predict
   LinearSVC.predict_proba
   LinearSVC.register
   LinearSVC.report
   LinearSVC.roc_curve
   LinearSVC.score
   LinearSVC.set_params
   LinearSVC.summarize
   LinearSVC.to_binary
   LinearSVC.to_memmodel
   LinearSVC.to_pmml
   LinearSVC.to_python
   LinearSVC.to_sql
   LinearSVC.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   LinearSVC.object_type
   LinearSVC.classes_

Logistic Regression
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica

.. autosummary::
   :toctree: api/

   linear_model.LogisticRegression

.. currentmodule:: verticapy.machine_learning.vertica.linear_model

**Methods:**

.. autosummary::
   :toctree: api/

   LogisticRegression.classification_report
   LogisticRegression.confusion_matrix
   LogisticRegression.contour
   LogisticRegression.cutoff_curve
   LogisticRegression.deploySQL
   LogisticRegression.does_model_exists
   LogisticRegression.drop
   LogisticRegression.export_models
   LogisticRegression.features_importance
   LogisticRegression.fit
   LogisticRegression.get_attributes
   LogisticRegression.get_match_index
   LogisticRegression.get_params
   LogisticRegression.get_plotting_lib
   LogisticRegression.get_vertica_attributes
   LogisticRegression.import_models
   LogisticRegression.lift_chart
   LogisticRegression.plot
   LogisticRegression.prc_curve
   LogisticRegression.predict
   LogisticRegression.predict_proba
   LogisticRegression.register
   LogisticRegression.report
   LogisticRegression.roc_curve
   LogisticRegression.score
   LogisticRegression.set_params
   LogisticRegression.summarize
   LogisticRegression.to_binary
   LogisticRegression.to_memmodel
   LogisticRegression.to_pmml
   LogisticRegression.to_python
   LogisticRegression.to_sql
   LogisticRegression.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   LogisticRegression.object_type
   LogisticRegression.classes_

_____

Tree-based algorithms
---------------------

Dummy Tree
~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   tree.DummyTreeClassifier

.. currentmodule:: verticapy.machine_learning.vertica.tree

**Methods:**

.. autosummary::
   :toctree: api/

   DummyTreeClassifier.classification_report
   DummyTreeClassifier.confusion_matrix
   DummyTreeClassifier.contour
   DummyTreeClassifier.cutoff_curve
   DummyTreeClassifier.deploySQL
   DummyTreeClassifier.does_model_exists
   DummyTreeClassifier.drop
   DummyTreeClassifier.export_models
   DummyTreeClassifier.features_importance
   DummyTreeClassifier.fit
   DummyTreeClassifier.get_attributes
   DummyTreeClassifier.get_match_index
   DummyTreeClassifier.get_params
   DummyTreeClassifier.get_plotting_lib
   DummyTreeClassifier.get_score
   DummyTreeClassifier.get_tree
   DummyTreeClassifier.get_vertica_attributes
   DummyTreeClassifier.import_models
   DummyTreeClassifier.lift_chart
   DummyTreeClassifier.plot
   DummyTreeClassifier.plot_tree
   DummyTreeClassifier.prc_curve
   DummyTreeClassifier.predict
   DummyTreeClassifier.predict_proba
   DummyTreeClassifier.register
   DummyTreeClassifier.report
   DummyTreeClassifier.roc_curve
   DummyTreeClassifier.score
   DummyTreeClassifier.set_params
   DummyTreeClassifier.summarize
   DummyTreeClassifier.to_binary
   DummyTreeClassifier.to_graphviz
   DummyTreeClassifier.to_memmodel
   DummyTreeClassifier.to_pmml
   DummyTreeClassifier.to_python
   DummyTreeClassifier.to_sql
   DummyTreeClassifier.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   DummyTreeClassifier.object_type


Decision Tree
~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   tree.DecisionTreeClassifier

.. currentmodule:: verticapy.machine_learning.vertica.tree

**Methods:**

.. autosummary::
   :toctree: api/


   DecisionTreeClassifier.classification_report
   DecisionTreeClassifier.confusion_matrix
   DecisionTreeClassifier.contour
   DecisionTreeClassifier.cutoff_curve
   DecisionTreeClassifier.deploySQL
   DecisionTreeClassifier.does_model_exists
   DecisionTreeClassifier.drop
   DecisionTreeClassifier.export_models
   DecisionTreeClassifier.features_importance
   DecisionTreeClassifier.fit
   DecisionTreeClassifier.get_attributes
   DecisionTreeClassifier.get_match_index
   DecisionTreeClassifier.get_params
   DecisionTreeClassifier.get_plotting_lib
   DecisionTreeClassifier.get_score
   DecisionTreeClassifier.get_tree
   DecisionTreeClassifier.get_vertica_attributes
   DecisionTreeClassifier.import_models
   DecisionTreeClassifier.lift_chart
   DecisionTreeClassifier.plot
   DecisionTreeClassifier.plot_tree
   DecisionTreeClassifier.prc_curve
   DecisionTreeClassifier.predict
   DecisionTreeClassifier.predict_proba
   DecisionTreeClassifier.register
   DecisionTreeClassifier.report
   DecisionTreeClassifier.roc_curve
   DecisionTreeClassifier.score
   DecisionTreeClassifier.set_params
   DecisionTreeClassifier.summarize
   DecisionTreeClassifier.to_binary
   DecisionTreeClassifier.to_graphviz
   DecisionTreeClassifier.to_memmodel
   DecisionTreeClassifier.to_pmml
   DecisionTreeClassifier.to_python
   DecisionTreeClassifier.to_sql
   DecisionTreeClassifier.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   DecisionTreeClassifier.object_type

Random Forest Classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   ensemble.RandomForestClassifier

.. currentmodule:: verticapy.machine_learning.vertica.ensemble

**Methods:**

.. autosummary::
   :toctree: api/


   RandomForestClassifier.classification_report
   RandomForestClassifier.confusion_matrix
   RandomForestClassifier.contour
   RandomForestClassifier.cutoff_curve
   RandomForestClassifier.deploySQL
   RandomForestClassifier.does_model_exists
   RandomForestClassifier.drop
   RandomForestClassifier.export_models
   RandomForestClassifier.features_importance
   RandomForestClassifier.fit
   RandomForestClassifier.get_attributes
   RandomForestClassifier.get_match_index
   RandomForestClassifier.get_params
   RandomForestClassifier.get_plotting_lib
   RandomForestClassifier.get_score
   RandomForestClassifier.get_tree
   RandomForestClassifier.get_vertica_attributes
   RandomForestClassifier.import_models
   RandomForestClassifier.lift_chart
   RandomForestClassifier.plot
   RandomForestClassifier.plot_tree
   RandomForestClassifier.prc_curve
   RandomForestClassifier.predict
   RandomForestClassifier.predict_proba
   RandomForestClassifier.register
   RandomForestClassifier.report
   RandomForestClassifier.roc_curve
   RandomForestClassifier.score
   RandomForestClassifier.set_params
   RandomForestClassifier.summarize
   RandomForestClassifier.to_binary
   RandomForestClassifier.to_graphviz
   RandomForestClassifier.to_memmodel
   RandomForestClassifier.to_pmml
   RandomForestClassifier.to_python
   RandomForestClassifier.to_sql
   RandomForestClassifier.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   RandomForestClassifier.object_type

XGBoost Classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   ensemble.XGBClassifier

.. currentmodule:: verticapy.machine_learning.vertica.ensemble

**Methods:**

.. autosummary::
   :toctree: api/


   XGBClassifier.classification_report
   XGBClassifier.confusion_matrix
   XGBClassifier.contour
   XGBClassifier.cutoff_curve
   XGBClassifier.deploySQL
   XGBClassifier.does_model_exists
   XGBClassifier.drop
   XGBClassifier.export_models
   XGBClassifier.features_importance
   XGBClassifier.fit
   XGBClassifier.get_attributes
   XGBClassifier.get_match_index
   XGBClassifier.get_params
   XGBClassifier.get_plotting_lib
   XGBClassifier.get_score
   XGBClassifier.get_tree
   XGBClassifier.get_vertica_attributes
   XGBClassifier.import_models
   XGBClassifier.lift_chart
   XGBClassifier.plot
   XGBClassifier.plot_tree
   XGBClassifier.prc_curve
   XGBClassifier.predict
   XGBClassifier.predict_proba
   XGBClassifier.register
   XGBClassifier.report
   XGBClassifier.roc_curve
   XGBClassifier.score
   XGBClassifier.set_params
   XGBClassifier.summarize
   XGBClassifier.to_binary
   XGBClassifier.to_graphviz
   XGBClassifier.to_json
   XGBClassifier.to_memmodel
   XGBClassifier.to_pmml
   XGBClassifier.to_python
   XGBClassifier.to_sql
   XGBClassifier.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   XGBClassifier.object_type

________

Naive Bayes
--------------

Naive Bayes
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   naive_bayes.NaiveBayes

.. currentmodule:: verticapy.machine_learning.vertica.naive_bayes

**Methods:**

.. autosummary::
   :toctree: api/


   NaiveBayes.classification_report
   NaiveBayes.confusion_matrix
   NaiveBayes.contour
   NaiveBayes.cutoff_curve
   NaiveBayes.deploySQL
   NaiveBayes.does_model_exists
   NaiveBayes.drop
   NaiveBayes.export_models
   NaiveBayes.fit
   NaiveBayes.get_attributes
   NaiveBayes.get_match_index
   NaiveBayes.get_params
   NaiveBayes.get_plotting_lib
   NaiveBayes.get_vertica_attributes
   NaiveBayes.import_models
   NaiveBayes.lift_chart
   NaiveBayes.prc_curve
   NaiveBayes.predict
   NaiveBayes.predict_proba
   NaiveBayes.register
   NaiveBayes.report
   NaiveBayes.roc_curve
   NaiveBayes.score
   NaiveBayes.set_params
   NaiveBayes.summarize
   NaiveBayes.to_binary
   NaiveBayes.to_memmodel
   NaiveBayes.to_pmml
   NaiveBayes.to_python
   NaiveBayes.to_sql
   NaiveBayes.to_tf

**Attributes:**

.. autosummary::
   :toctree: api/

   NaiveBayes.object_type
_______

Neighbors
-----------

K-Nearest Neighbors Classifier (Beta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   neighbors.KNeighborsClassifier

.. currentmodule:: verticapy.machine_learning.vertica.neighbors

**Methods:**

.. autosummary::
   :toctree: api/


   KNeighborsClassifier.classification_report
   KNeighborsClassifier.confusion_matrix
   KNeighborsClassifier.contour
   KNeighborsClassifier.cutoff_curve
   KNeighborsClassifier.deploySQL
   KNeighborsClassifier.does_model_exists
   KNeighborsClassifier.drop
   KNeighborsClassifier.export_models
   KNeighborsClassifier.fit
   KNeighborsClassifier.get_attributes
   KNeighborsClassifier.get_match_index
   KNeighborsClassifier.get_params
   KNeighborsClassifier.get_plotting_lib
   KNeighborsClassifier.get_vertica_attributes
   KNeighborsClassifier.import_models
   KNeighborsClassifier.lift_chart
   KNeighborsClassifier.prc_curve
   KNeighborsClassifier.predict
   KNeighborsClassifier.predict_proba
   KNeighborsClassifier.register
   KNeighborsClassifier.report
   KNeighborsClassifier.roc_curve
   KNeighborsClassifier.score
   KNeighborsClassifier.set_params
   KNeighborsClassifier.summarize
   KNeighborsClassifier.to_binary
   KNeighborsClassifier.to_pmml
   KNeighborsClassifier.to_python
   KNeighborsClassifier.to_sql
   KNeighborsClassifier.to_tf


**Attributes:**

.. autosummary::
   :toctree: api/

   KNeighborsClassifier.object_type


Nearest Centroid (Beta)
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: verticapy.machine_learning.vertica
   
.. autosummary::
   :toctree: api/

   cluster.NearestCentroid

.. currentmodule:: verticapy.machine_learning.vertica.cluster

**Methods:**

.. autosummary::
   :toctree: api/


   NearestCentroid.classification_report
   NearestCentroid.confusion_matrix
   NearestCentroid.contour
   NearestCentroid.cutoff_curve
   NearestCentroid.deploySQL
   NearestCentroid.does_model_exists
   NearestCentroid.drop
   NearestCentroid.export_models
   NearestCentroid.fit
   NearestCentroid.get_attributes
   NearestCentroid.get_match_index
   NearestCentroid.get_params
   NearestCentroid.get_plotting_lib
   NearestCentroid.get_vertica_attributes
   NearestCentroid.import_models
   NearestCentroid.lift_chart
   NearestCentroid.prc_curve
   NearestCentroid.predict
   NearestCentroid.predict_proba
   NearestCentroid.register
   NearestCentroid.report
   NearestCentroid.roc_curve
   NearestCentroid.score
   NearestCentroid.set_params
   NearestCentroid.summarize
   NearestCentroid.to_binary
   NearestCentroid.to_memmodel
   NearestCentroid.to_pmml
   NearestCentroid.to_python
   NearestCentroid.to_sql
   NearestCentroid.to_tf


**Attributes:**

.. autosummary::
   :toctree: api/

   NearestCentroid.object_type


_____

