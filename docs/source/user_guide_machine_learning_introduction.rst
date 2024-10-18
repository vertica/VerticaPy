.. _user_guide.machine_learning.introduction:

=================================
Introduction to Machine Learning
=================================

One of the last stages of the data science life cycle is the Data Modeling. Machine learning algorithms are a set of statistical techniques that build mathematical models from training data. These algorithms come in two types:
- **Supervised:** these algorithms are used when we want to predict a response column.
- **Unsupervised:** these algorithms are used when we want to detect anomalies or when we want to segment the data. No response column is needed.

Supervised Learning
--------------------

Supervised Learning techniques map an input to an output based on some example dataset. This type of learning consists of two main types:
- **Regression:** The Response is numerical (``Linear Regression``, ``SVM Regression``, ``RF Regression``...).
- **Classification:** The Response is categorical (``Gradient Boosting``, ``Naive Bayes``, ``Logistic Regression``...).
For example, predicting the total charges of a Telco customer using their tenure would be a type of regression. The following code is drawing a linear regression using the 'TotalCharges' as a function of the 'tenure' in the `telco churn dataset <https://github.com/vertica/VerticaPy/tree/master/docs/source/notebooks/data_exploration/correlations/data>`_.

.. code-block:: python

    import verticapy as vp

    churn = vp.read_csv("churn.csv")

    from verticapy.machine_learning.vertica import LinearRegression

    model = LinearRegression()
    model.fit(churn, ["tenure"], "TotalCharges")
    model.plot()

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp

    churn = vp.read_csv("/project/data/VerticaPy/docs/source/_static/website/user_guides/data_exploration/churn.csv")

    from verticapy.machine_learning.vertica import LinearRegression

    model = LinearRegression()
    model.fit(churn, ["tenure"], "TotalCharges")
    fig = model.plot()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_ml_plot_introduction_1.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_plot_introduction_1.html

In contrast, when we have to predict a categorical column, we're dealing with classification.

In the following example, we use a ``Linear Support Vector Classification`` (SVC) to predict the species of a flower based on its petal and sepal lengths.

.. code-block::

    from verticapy.datasets import load_iris

    iris = load_iris()
    iris.one_hot_encode()

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.datasets import load_iris

    iris = load_iris()
    res = iris.one_hot_encode()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_ml_table_introduction_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
  :file: /project/data/VerticaPy/docs/figures/ug_ml_table_introduction_1.html

.. code-block:: python

    from verticapy.machine_learning.vertica import LinearSVC

    model = LinearSVC()
    model.fit(iris, ["PetalLengthCm", "SepalLengthCm"], "Species_Iris-setosa")
    model.plot()

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import LinearSVC

    model = LinearSVC()
    model.fit(iris, ["PetalLengthCm", "SepalLengthCm"], "Species_Iris-setosa")
    fig = model.plot()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_ml_plot_introduction_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_plot_introduction_2.html

When we have more than two categories, we use the expression 'Multiclass Classification' instead of 'Classification'.

Unsupervised Learning
----------------------

These algorithms are to used to segment the data (``k-means``, ``DBSCAN``, etc.) or to detect anomalies (``Local Outlier Factor``, ``Z-Score`` Techniques...). In particular, they're useful for finding patterns in data without labels. For example, let's use a k-means algorithm to create different clusters on the Iris dataset. Each cluster will represent a flower's species.

.. code-block:: python

    from verticapy.machine_learning.vertica import KMeans

    model = KMeans(n_cluster = 3)
    model.fit(iris, ["PetalLengthCm", "SepalLengthCm"])
    model.plot()

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import KMeans

    model = KMeans(n_cluster = 3)
    model.fit(iris, ["PetalLengthCm", "SepalLengthCm"])
    fig = model.plot()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_ml_plot_introduction_3.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_plot_introduction_3.html

In this section, we went over a few of the many ML algorithms available in VerticaPy. In the next lesson, we'll cover creating a regression model.