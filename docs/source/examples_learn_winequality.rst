.. _examples.learn.winequality:

Wine Quality
=============

This example uses the Wine Quality dataset to predict the quality of white wine. 
You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/learn/winequality/winequality.ipynb>`_.

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- total sulfur dioxide
- free sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality (score between 0 and 10)

We will follow the data science cycle (Data Exploration - Data Preparation - Data Modeling - Model Evaluation - Model Deployment) to solve this problem.

Initialization
----------------

This example uses the following version of VerticaPy:

.. ipython:: python
    
    import verticapy as vp

    vp.__version__

Connect to Vertica. This example uses an existing connection called "VerticaDSN." 
For details on how to create a connection, see the :ref:`connection` tutorial.
You can skip the below cell if you already have an established connection.

.. code-block:: python
    
    vp.connect("VerticaDSN")

Let's create a Virtual DataFrame of the dataset.

.. code-block:: python

    from verticapy.datasets import load_winequality
    
    winequality = load_winequality()
    winequality.head(5)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_winequality
    winequality = load_winequality()
    res = winequality.head(5)
    html_file = open("SPHINX_DIRECTORY/figures/examples_winequality_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_winequality_table_head.html

Data Exploration and Preparation
----------------------------------

Let's explore the data by displaying descriptive statistics of all the columns.

.. code-block:: python

    winequality.describe()

.. ipython:: python
    :suppress:

    res = winequality.describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_winequality_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_winequality_table_describe.html

The quality of a wine is based on the equilibrium between certain components:
 - **For red wines:** tannin/smoothness/acidity
 - **For white wines:** smoothness/acidity
 
Based on this, we don't have the data to create a good model for red wines (the tannins weren't extracted). 
We do, however, have enough data to make a good model for white wines, so let's filter out red wines from our study.

.. code-block:: python

    winequality.filter(winequality["color"] == "white").drop(["good", "color"])

.. ipython:: python
    :suppress:

    winequality.filter(winequality["color"] == "white").drop(["good", "color"])
    res = winequality
    html_file = open("SPHINX_DIRECTORY/figures/examples_winequality_table_filter.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_winequality_table_filter.html

Let's draw the correlation matrix of the dataset.

.. code-block:: python

    winequality.corr(method = "spearman")

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = winequality.corr(method = "spearman", width = 800, height = 800)
    fig.write_html("SPHINX_DIRECTORY/figures/examples_winequality_table_corr_matrix.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_winequality_table_corr_matrix.html

We can see a strong correlation between the density and the alcohol degree (the alcohol degree describes the density of pure ethanol in the wine).

We can drop the ``density`` column since it doesn't influence the quality of the white wine (instead, its presence will just bias the data).

.. code-block:: python

    winequality.drop(["density"])

.. ipython:: python
    :suppress:

    winequality.drop(["density"])
    res = winequality
    html_file = open("SPHINX_DIRECTORY/figures/examples_winequality_table_drop.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_winequality_table_drop.html

We're working with the scores given by wine tasters, so it's likely that two closely competing wines will have a similar score. Knowing this, a ``k-nearest neighbors`` (KNN) model would be best.

KNN is sensitive to unnormalized data so we'll have to normalize our data.

.. code-block:: python

    winequality.normalize(
        [
            "free_sulfur_dioxide", 
            "residual_sugar", 
            "pH", 
            "sulphates", 
            "volatile_acidity", 
            "fixed_acidity",
            "citric_acid",
            "chlorides",
            "total_sulfur_dioxide",
            "alcohol"
        ],
        method = "robust_zscore",
    )


.. ipython:: python
    :suppress:

    winequality.normalize(
        [
            "free_sulfur_dioxide", 
            "residual_sugar", 
            "pH", 
            "sulphates", 
            "volatile_acidity", 
            "fixed_acidity",
            "citric_acid",
            "chlorides",
            "total_sulfur_dioxide",
            "alcohol"
        ],
        method = "robust_zscore",
    )
    res = winequality
    html_file = open("SPHINX_DIRECTORY/figures/examples_winequality_table_normalize.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_winequality_table_normalize.html

Machine Learning
-----------------

Let's create our ``KNN`` model.

.. code-block:: python

    from verticapy.machine_learning.vertica import KNeighborsRegressor
    from verticapy.machine_learning.model_selection import cross_validate

    predictors = winequality.get_columns(exclude_columns = ["quality"])
    model = KNeighborsRegressor(name = "winequality_KNN", n_neighbors = 50)
    cross_validate(model, winequality, predictors, "quality")

.. ipython:: python
    :suppress:

    from verticapy.machine_learning.vertica import KNeighborsRegressor
    from verticapy.machine_learning.model_selection import cross_validate

    predictors = winequality.get_columns(exclude_columns = ["quality"])
    model = KNeighborsRegressor(name = "winequality_KNN", n_neighbors = 50)
    res = cross_validate(model, winequality, predictors, "quality")
    html_file = open("SPHINX_DIRECTORY/figures/examples_winequality_table_ml_cv.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_winequality_table_ml_cv.html

Our model is pretty good. Our predicted scores have a median absolute error of less than 0.5. 
If we want to improve this model, we'll probably need more relevant features.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!