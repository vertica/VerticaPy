.. _user_guide.machine_learning.classification:

===============
Classification
===============

Classifications are ML algorithms used to predict categorical response columns. For predicting more than two categories, these are called ``Multiclass Classifications``. Examples of classification are predicting the flower species using specific characteristics or predicting whether Telco customers will churn.

To understand how to create a classification model, let's predict the species of flowers with the Iris dataset.

We'll start by importing the ``Random Forest Classifier``.

.. ipython:: python

    from verticapy.machine_learning.vertica import RandomForestClassifier

Next, we'll create a model object.

.. ipython:: python

    model = RandomForestClassifier()

Let's use the iris dataset.

.. ipython:: python

    from verticapy.datasets import load_iris

    iris = load_iris()

Now that the data is loaded, we can fit the model.

.. ipython:: python
    :okwarning:

    model.fit(iris, ["PetalLengthCm", "SepalLengthCm"], "Species")

We have many metrics to evaluate the model.

.. code-block::

    model.report()

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.report()
    html_file = open("SPHINX_DIRECTORY/figures/ug_ml_table_classification_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_ml_table_classification_1.html

You can add the predictions to your dataset.

.. code-block::

    model.predict(iris, name = "prediction")

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.predict(iris, name = "prediction")
    html_file = open("SPHINX_DIRECTORY/figures/ug_ml_table_classification_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_ml_table_classification_2.html

You can also add the probabilities.

.. code-block::

    model.predict_proba(iris, name = "prob")

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.predict_proba(iris, name = "prob")
    html_file = open("SPHINX_DIRECTORY/figures/ug_ml_table_classification_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_ml_table_classification_3.html

Our example forgoes splitting the data into training and testing, which is important for real-world work. Our main goal in this lesson is to look at the metrics used to evaluate classifications. The most famous metric is accuracy: generally speaking, the closer accuracy is to 1, the better the model is. However, taking metrics at face value can lead to incorrect interpretations.

For example, let's say our goal is to identify bank fraud. Fraudulent activity is relatively rare, so let's say that they represent less than ``1%`` of the data. If we were to predict that there are no frauds in the dataset, we'd end up with an accuracy of ``99%``. This is why ROC ``AUC`` and PRC ``AUC`` are more robust metrics.

That said, a good model is simply a model that might solve a the given problem. In that regard, any model is better than a random one.

In the next lesson, we'll go over :ref:`user_guide.machine_learning.time_series`