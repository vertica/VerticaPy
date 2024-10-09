.. _examples.iris:

Iris
=====

This example uses the 'iris' dataset to predict the species of various flowers based on their physical features. You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/learn/iris/iris.ipynb>`_.

- **PetalLengthCm:** Petal Length in cm
- **PetalWidthCm:** Petal Width in cm
- **SepalLengthCm:** Sepal Length in cm
- **SepalWidthCm:** Sepal Width in cm
- **Species:** The Flower Species (Setosa, Virginica, Versicolor)

We will follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem.

Initialization
---------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp

    vp.__version__

Connect to Vertica. This example uses an existing connection called "VerticaDSN". 
For details on how to create a connection, see the :ref:`connection` tutorial.

You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Let's create a Virtual DataFrame of the dataset.

.. code-block:: python

    from verticapy.datasets import load_iris

    iris = load_iris()
    iris.head(5)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_iris
    iris = load_iris()
    res = iris.head(5)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_iris_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_iris_table_head.html

Data Exploration and Preparation
---------------------------------

Let's explore the data by displaying descriptive statistics of all the columns.

.. code-block:: python

    iris.describe(method = "categorical", unique = True)

.. ipython:: python
    :suppress:

    res = iris.describe(method = "categorical", unique = True)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_iris_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_iris_table_describe.html

We don't have much data here, but that's okay; since different flower species have different proportions and ratios between those proportions, we can start by making ratios between each feature.

We'll need to use the One-Hot Encoder on the 'Species' to get information about each species.

.. code-block:: python
    
    iris["Species"].one_hot_encode(drop_first = False)
    iris["ratio_pwl"] = iris["PetalWidthCm"] / iris["PetalLengthCm"]
    iris["ratio_swl"] = iris["SepalWidthCm"] / iris["SepalLengthCm"]

.. ipython:: python
    :suppress:
    
    iris["Species"].one_hot_encode(drop_first = False)
    iris["ratio_pwl"] = iris["PetalWidthCm"] / iris["PetalLengthCm"]
    iris["ratio_swl"] = iris["SepalWidthCm"] / iris["SepalLengthCm"]

We can draw the correlation matrix (Pearson correlation coefficient) of the new features to see if there are some linear links.

.. code-block:: python

    iris.corr()

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = iris.corr(width = 800, height = 800)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_iris_table_corr_matrix.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_iris_table_corr_matrix.html

The Iris setosa is highly linearly correlated with the petal length and the sepal ratio. We can see a perfect separation using the two features (though we can also see this separation the petal length alone).

.. code-block:: python

    iris.scatter(
        columns = ["PetalLengthCm", "ratio_swl"], 
        by = "Species",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    fig = iris.scatter(
        columns = ["PetalLengthCm", "ratio_swl"], 
        by = "Species",
        width = 800,
        height = 800,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_iris_scatter_1.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_iris_scatter_1.html

We can we a clear linear separation between the Iris setosa and the other species, but we'll need more features to identify the differences between Iris virginica and Iris versicolor.

.. code-block:: python

    iris.scatter(
        columns = [
            "PetalLengthCm", 
            "PetalWidthCm", 
            "SepalLengthCm",
        ], 
        by = "Species",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    fig = iris.scatter(
        columns = [
            "PetalLengthCm", 
            "PetalWidthCm", 
            "SepalLengthCm",
        ],
        by = "Species",
        width = 800,
        height = 800,
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_iris_scatter_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_iris_scatter_2.html

Our strategy is simple: we'll use two Linear Support Vector Classification (SVC): one to classify the Iris setosa and another to classify the Iris versicolor.

Machine Learning
-----------------

Let's build the first ``LinearSVC`` to predict if a flower is an Iris setosa.

.. code-block:: python

    from verticapy.machine_learning.vertica import LinearSVC
    from verticapy.machine_learning.model_selection import cross_validate

    predictors = ["PetalLengthCm", "ratio_swl"]
    response = "Species_Iris-setosa"
    model = LinearSVC("svc_setosa_iris")
    cross_validate(model, iris, predictors, response)

.. ipython:: python
    :suppress:

    from verticapy.machine_learning.vertica import LinearSVC
    from verticapy.machine_learning.model_selection import cross_validate

    predictors = ["PetalLengthCm", "ratio_swl"]
    response = "Species_Iris-setosa"
    model = LinearSVC("svc_setosa_iris")
    res = cross_validate(model, iris, predictors, response)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_iris_table_ml_cv.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_iris_table_ml_cv.html

Our model is excellent. Let's build it using the entire dataset.

.. ipython:: python
    
    model.fit(iris, predictors, response)

Let's plot the model to see the perfect separation.

.. code-block:: python

    model.plot()

.. ipython:: python
    :suppress:
    :okwarning:

    fig = model.plot(width = 800, height = 800)
    fig.write_html("/project/data/VerticaPy/docs/figures/examples_model_plot.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_model_plot.html

We can add this probability to the ``vDataFrame``.

.. code-block:: python

    model.predict_proba(iris, name = "setosa", pos_label = 1)

.. ipython:: python
    :suppress:

    res = model.predict_proba(iris, name = "setosa", pos_label = 1)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_model_predict_proba.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_model_predict_proba.html

Let's create a model to classify the Iris virginica.

.. code-block:: python

    predictors = [
        "PetalLengthCm",
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalWidthCm",
        "ratio_pwl",
        "ratio_swl",
    ]
    response = "Species_Iris-virginica"
    model = LinearSVC("svc_virginica_iris")
    cross_validate(model, iris, predictors, response)

.. ipython:: python
    :suppress:

    predictors = [
        "PetalLengthCm",
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalWidthCm",
        "ratio_pwl",
        "ratio_swl",
    ]
    response = "Species_Iris-virginica"
    model = LinearSVC("svc_virginica_iris")
    res = cross_validate(model, iris, predictors, response)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_iris_table_ml_cv_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_iris_table_ml_cv_2.html

We have another excellent model. Let's add it to the ``vDataFrame``.

.. code-block:: python

    model.fit(iris, predictors, response)
    model.predict_proba(iris, name = "virginica", pos_label = 1)

.. ipython:: python
    :suppress:

    model.fit(iris, predictors, response)
    res = model.predict_proba(iris, name = "virginica", pos_label = 1)
    html_file = open("/project/data/VerticaPy/docs/figures/examples_model_predict_proba_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/examples_model_predict_proba_2.html

Let's evaluate our final model (the combination of two ``LinearSVC``s).

.. code-block:: python

    iris.case_when(
        "prediction",
        iris["setosa"] > 0.5, 'Iris-setosa',
        iris["virginica"] > 0.5, 'Iris-virginica',
        'Iris-versicolor',
    )
    iris["score"] = (iris["Species"] == iris["prediction"])

.. ipython:: python
    :suppress:

    iris.case_when(
        "prediction",
        iris["setosa"] > 0.5, 'Iris-setosa',
        iris["virginica"] > 0.5, 'Iris-virginica',
        'Iris-versicolor',
    )
    iris["score"] = (iris["Species"] == iris["prediction"])

.. ipython:: python

    iris["score"].avg()

We have a great model with an accuracy of 96% on an entirely balanced dataset.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!