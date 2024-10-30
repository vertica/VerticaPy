.. _user_guide.machine_learning.model_tracking:

==============================
Model Tracking and Versioning
==============================

Introduction
-------------

VerticaPy is an open-source Python package on top of Vertica database that supports pandas-like virtual dataframes over database relations. VerticaPy provides scikit-type machine learning functionality on these virtual dataframes. Data is not moved out of the database while performing machine learning or statistical analysis on virtual dataframes. Instead, the computations are done at scale in a distributed fashion inside the Vertica cluster. VerticaPy also takes advantage of multiple Python libraries to create a variety of charts, providing a quick and easy method to illustrate your statistical data.

In this article, we will introduce two new MLOps tools recently added to VerticaPy: Model Tracking and Model Versioning.

Model Tracking
---------------

Data scientists usually train many ML models for a project. To help choose the best model, data scientists need a way to keep track of all candidate models and compare them using various metrics. VerticaPy provides a model tracking system to facilitate this process for a given experiment. The data scientist first creates an experiment object and then adds candidate models to that experiment. The information related to each experiment can be automatically backed up in the database, so if the Python environment is closed for any reason, like a holiday, the data scientist has peace of mind that the experiment can be easily retrieved. The experiment object also provides methods to easily compare the prediction performance of its associated models and to pick the model with the best performance on a specific test dataset.

The following example demonstrates how the model tracking feature can be used for an experiment that trains a few binary-classifier models on the Titanic dataset. First, we must load the titanic data into our database and store it as a virtual dataframe (vDF):

.. ipython:: python
    :okwarning:

    from verticapy.datasets import load_titanic

    titanic_vDF = load_titanic()
    predictors = ["age", "fare", "pclass"]
    response = "survived"

We then define a :py:func:`~verticapy.mlops.model_tracking.vExperiment` object to track the candidate models. To define the experiment object, specify the following parameters:

- experiment_name: The name of the experiment.
- test_relation: Relation or vDF to use to test the model.
- X: List of the predictors.
- y: Response column.

.. note:: If experiments_type is set to clustering, test_relation, X, and Y must be set to None.

The following parameters are optional:

- experiment_type: By default ``auto``, meaning VerticaPy tries to detect the experiment type from the response value. However, it might be cleaner to explicitly specify the experiment type. The other valid values for this parameter are ``regressor`` (for regression models), ``binary`` (for binary classification models), ``multi`` (for multiclass classification models), and ``clustering`` (for clustering models).
- experiment_table: The name of the table ([schema_name.]table_name) in the database to archive the experiment. The experiment information won't be backed up in the database without specifying this parameter. If the table already exists, its previously stored experiments are loaded to the object. In this case, the user must have ``SELECT``, ``INSERT``, and ``DELETE`` privileges on the table. If the table doesn't exist and the user has the necessary privileges for creating such a table, the table is created.

.. ipython:: python
    :okwarning:

    import verticapy.mlops.model_tracking as mt

    my_experiment_1 = mt.vExperiment(
        experiment_name = "my_exp_1",
        test_relation = titanic_vDF,
        X=predictors,
        y=response,
        experiment_type="binary",
        experiment_table="my_exp_table_1",
    )

After creating the experiment object, we can train different models and add them to the experiment:

.. ipython:: python
    :okwarning:

    # training a LogisticRegression model
    from verticapy.machine_learning.vertica import LogisticRegression

    model_1 = LogisticRegression("logistic_reg_m", overwrite_model = True)
    model_1.fit(titanic_vDF, predictors, response)
    my_experiment_1.add_model(model_1)

    # training a LinearSVC model
    from verticapy.machine_learning.vertica import LinearSVC

    model_2 = LinearSVC("svc_m", overwrite_model = True)
    model_2.fit(titanic_vDF, predictors, response)
    my_experiment_1.add_model(model_2)

    # training a DecisionTreeClassifier model
    from verticapy.machine_learning.vertica import DecisionTreeClassifier

    model_3 = DecisionTreeClassifier("tree_m", overwrite_model = True, max_depth = 3)
    model_3.fit(titanic_vDF, predictors, response)
    my_experiment_1.add_model(model_3)

So far we have only added three models to the experiment, but we could add many more in a real scenario. Using the experiment object, we can easily list the models in the experiment and pick the one with the best prediction performance based on a specified metric.

.. code-block:: python

    my_experiment_1.list_models()

.. ipython:: python
    :suppress:
    :okwarning:

    res = my_experiment_1.list_models()
    html_file = open("SPHINX_DIRECTORY/figures/ug_ml_model_tracking_list_models.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_ml_model_tracking_list_models.html

.. ipython:: python

    top_model = my_experiment_1.load_best_model(metric = "auc")

The experiment object facilitates not only model tracking but also makes cleanup super easy, especially in real-world 
scenarios where there is often a large number of leftover models. The :py:func:`~verticapy.machine_learning.vertica.LogisticRegression.drop` method drops from the database the info of the experiment and all associated models other than those specified in the keeping_models list.

.. ipython:: python
    :okwarning:

    my_experiment_1.drop(keeping_models=[top_model.model_name])

Experiments are also helpful for performing grid search on hyper-parameters. The following example shows how they can 
be used to study the impact of the max_iter parameter on the prediction performance of :py:mod:`~verticapy.machine_learning.vertica.linear_model.LogisticRegression` models.

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    vp.set_option("plotting_lib", "matplotlib")

.. ipython:: python
    :okwarning:

    # creating an experiment
    my_experiment_2 = mt.vExperiment(
        experiment_name = "my_exp_2",
        test_relation = titanic_vDF,
        X = predictors,
        y = response,
        experiment_type = "binary",
    )

    # training LogisticRegression with different values of max_iter
    for i in range(1, 5):
        model = LogisticRegression(max_iter = i)
        model.fit(titanic_vDF, predictors, response)
        my_experiment_2.add_model(model)
        
    # plotting prc_auc vs max_iter
    @savefig my_experiment_2_plot_max_iter_prc.png
    my_experiment_2.plot("max_iter", "prc_auc")

    # cleaning all the models associated to the experiment from the database
    my_experiment_2.drop()

Model Versioning
-----------------

In Vertica version 12.0.4, we added support for In-DB ML Model Versioning. Now, we have integrated it into VerticaPy so that users can utilize its capabilities along with the other tools in VerticaPy. In VerticaPy, model versioning is a wrapper around an SQL API already built in Vertica. For more information about the concepts of model versioning in Vertica, see the Vertica documentation.

To showcase model versioning, we will begin by registering the ``top_model`` picked from the above experiment.

.. ipython:: python
    :okwarning:

    top_model.register("top_model_demo")

When the model owner registers the model, its ownership changes to ``DBADMIN``, and the previous owner receives ``USAGE`` privileges. Registered models are referred to by their registered_name and version. Only ``DBADMIN`` or a user with the ``MLSUPERVISOR`` role can change the status of a registered model. We have provided the :py:func:`~verticapy.mlops.model_versioning.RegisteredModel` class in VerticaPy for working with registered models.

We will now make a :py:func:`~verticapy.mlops.model_versioning.RegisteredModel` object for our recently registered model and change its status to ``production``. We can then use the registered model for scoring.

.. ipython:: python

    import verticapy.mlops.model_versioning as mv

    rm = mv.RegisteredModel("top_model_demo")

To see the list of all models registered as ``top_model_demo``, use the :py:func:`~verticapy.mlops.model_versioning.RegisteredModel.list_models` method.

.. code-block:: python

    rm.list_models()

.. ipython:: python
    :suppress:
    :okwarning:

    res = rm.list_models()
    html_file = open("SPHINX_DIRECTORY/figures/ug_ml_model_tracking_list_models_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_ml_model_tracking_list_models_2.html

The model we just registered has a status of ``under_review``. The next step is to change the status of the model to ``staging``, which is meant for A/B testing the model. Assuming the model performs well, we will promote it to the "production" status. Please note that we should specify the right version of the registered model from the above table.

.. ipython:: python
    :okwarning:

    # Getting the current version
    version = rm.list_models()["registered_version"][0]

    # changing the status of the model to staging
    rm.change_status(version = version, new_status = "staging")

    # changing the status of the model to production
    rm.change_status(version = version, new_status = "production")

There can only be one version of the registered model in ``production`` at any time. The following predict function applies to the model with ``production`` status by default. 

If you want to run the predict function on a model with a status other than "production", you must also specify the model version.

.. code-block:: python

    rm.predict(
        titanic_vDF,
        X = predictors,
        name = "predicted_value",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    res = rm.predict(titanic_vDF, X = predictors, name = "predicted_value")
    html_file = open("SPHINX_DIRECTORY/figures/ug_ml_model_tracking_predict.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_ml_model_tracking_predict.html

``DBADMIN`` and users who are granted ``SELECT`` privileges on the ``v_monitor.model_status_history`` table are able to monitor the status history of registered models.

.. code-block:: python

    rm.list_status_history()

.. ipython:: python
    :suppress:
    :okwarning:

    res = rm.list_status_history()
    html_file = open("SPHINX_DIRECTORY/figures/ug_ml_model_tracking_list_status_history.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_ml_model_tracking_list_status_history.html

Conclusion
-----------

The addition of model tracking and model versioning to the VerticaPy toolkit greatly improves VerticaPy's MLOps capabilities. We are constantly working to improve VerticaPy and address the needs of data scientists who wish to harness the power of Vertica database to empower their data analyses. If you have any comments or questions, don't hesitate to reach out in the VerticaPy github community.

This concludes the fundamental lessons on machine learning algorithms in VerticaPy.