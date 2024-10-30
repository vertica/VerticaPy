.. _examples.titanic:

Titanic
========

This example uses the ``titanic`` dataset to predict the survival of passengers on the Titanic. You can download the Jupyter Notebook of the study `here <https://github.com/vertica/VerticaPy/blob/master/examples/learn/titanic/titanic.ipynb>`_.

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

    from verticapy.datasets import load_titanic

    titanic = load_titanic()
    titanic.head(5)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_titanic
    titanic = load_titanic()
    res = titanic.head(5)
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_head.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_head.html

Data Exploration and Preparation
---------------------------------

Let's explore the data by displaying descriptive statistics of all the columns.

.. code-block:: python

    titanic.describe(method = "categorical", unique = True)

.. ipython:: python
    :suppress:

    res = titanic.describe(method = "categorical", unique = True)
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_describe.html

The columns "body" (passenger ID), "home.dest" (passenger origin/destination), "embarked" (origin port) and "ticket" (ticket ID) shouldn't influence survival, so we can ignore these.

Let's focus our analysis on the columns "name" and "cabin". We'll begin with the passengers' names.

.. code-block:: python

    from verticapy.machine_learning.vertica import CountVectorizer

    model = CountVectorizer()
    model.fit(titanic, ["Name"])
    model.transform()

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import CountVectorizer

    model = CountVectorizer()
    model.fit(titanic, ["Name"])
    res = model.transform()
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_count_vect_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_count_vect_1.html

Passengers' titles might come in handy. We can extract these from their names.

Let's move on to the cabins.

.. code-block:: python

    model = CountVectorizer()
    model.fit("titanic", ["cabin"])
    model.transform()

.. ipython:: python
    :suppress:
    :okwarning:

    model = CountVectorizer()
    model.fit("titanic", ["cabin"])
    res = model.transform()
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_count_vect_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_count_vect_2.html

Here, we have the cabin IDs, the letter of which represents a certain position on the boat. Let's see how often each cabin occurs in the dataset.

.. code-block:: python

    model = CountVectorizer()
    model.fit("titanic", ["cabin"])
    model.transform()["token"].str_slice(1, 1).groupby(
        columns = ["token"], expr = ["SUM(cnt)"]
    ).head(30)

.. ipython:: python
    :suppress:
    :okwarning:

    model = CountVectorizer()
    model.fit("titanic", ["cabin"])
    res = model.transform()["token"].str_slice(1, 1).groupby(
        columns = ["token"], expr = ["SUM(cnt)"]
    ).head(30)
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_count_vect_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_count_vect_3.html

While NULL values for "boat" clearly represent passengers who have a dedicated "lifeboat", we can't be so sure about ``NULL`` values for "cabin". We can guess that these might represent passengers without a cabin. If this is the case, then these are missing values not at random (MNAR).

We'll revisit this problem later. For now, let's drop the columns that don't affect survival and then encode the rest.

.. code-block:: python

    titanic.drop(["body", "home.dest", "embarked", "ticket"])
    titanic["cabin"].str_slice(1, 1)["name"].str_extract(
            ' ([A-Za-z]+)\.')["boat"].fillna(
            method = "0ifnull"
    )["cabin"].fillna("No Cabin")

.. ipython:: python
    :suppress:
    :okwarning:

    titanic.drop(["body", "home.dest", "embarked", "ticket"])
    res = titanic["cabin"].str_slice(1, 1)["name"].str_extract(
            ' ([A-Za-z]+)\.')["boat"].fillna(
            method = "0ifnull"
    )["cabin"].fillna("No Cabin")
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean.html

Looking at our data now, we can see that some first class passengers have a ``NULL`` value for their cabin, so we can safely say that our assumption about the meaning of a ``NULL`` value of "cabin" turned out to be incorrect. This means that the "cabin" column has far too many missing values at random (MAR). We'll have to drop it.

.. code-block:: python

    titanic["cabin"].drop()

.. ipython:: python
    :suppress:

    res = titanic["cabin"].drop()
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean_2.html

Let's look at descriptive statistics of the entire Virtual Dataframe.

.. code-block:: python

    titanic.describe(method = "all")

.. ipython:: python
    :suppress:

    res = titanic.describe(method = "all")
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_describe_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_describe_2.html

Descriptive statistics can give us valuable insights into our data. Notice, for example, that the column "fare" has many outliers (The maximum of 512.33 is much greater than the 9th decile of 79.13). Most passengers traveled in 3rd class (median of pclass = 3).

The "sibsp" column represents the number of siblings for each passenger, while the "parch" column represents the number of parents and children. We can use these to create a new feature: "family_size".

.. ipython:: python

    titanic["family_size"] = titanic["parch"] + titanic["sibsp"] + 1

Let's move on to outliers. We have several tools for locating outliers (:py:mod:`~verticapy.machine_learning.vertica.LocalOutlierFactor`, :py:mod:`~verticapy.machine_learning.vertica.DBSCAN`, :py:mod:`~verticapy.machine_learning.vertica.cluster.KMeans`...), but we'll just use winsorization in this example. Again, "fare" has many outliers, so we'll start there.

.. code-block:: python

    titanic["fare"].fill_outliers(
        method = "winsorize", 
        alpha = 0.03,
    )

.. ipython:: python
    :suppress:
    :okwarning:

    res = titanic["fare"].fill_outliers(
        method = "winsorize", 
        alpha = 0.03,
    )
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean_3.html

Let's encode the column ``sex`` so we can use it with numerical methods.

.. code-block:: python

    titanic["sex"].label_encode()

.. ipython:: python
    :suppress:
    :okwarning:

    res = titanic["sex"].label_encode()
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean_4.html

The column ``age`` has too many missing values and since most machine learning algorithms can't handle missing values, we need to impute our data. Let's fill the missing values using the average ``age`` of the passengers that have the same "pclass" and "sex".

.. code-block:: python

    titanic["age"].fillna(method = "mean", by = ["pclass", "sex"])

.. ipython:: python
    :suppress:
    :okwarning:

    res = titanic["age"].fillna(method = "mean", by = ["pclass", "sex"])
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean_5.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_drop_clean_5.html

Let's draw the correlation matrix to see the links between variables.

.. code-block:: python

    titanic.corr(method = "spearman")

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = titanic.corr(method = "spearman", width = 800, height = 800)
    fig.write_html("SPHINX_DIRECTORY/figures/examples_titanic_table_corr_matrix.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_corr_matrix.html

Fare correlates strongly with family size. This is about what you would expect: a larger family means more tickets, and more tickets means a greater fare.

Survival correlates strongly with whether or not a passenger has a lifeboat (the ``boat`` variable). Still, to increase the generality of our model, we should avoid predictions based on just one variable. Let's split the study into two use cases:

- Passengers with a lifeboat
- Passengers without a lifeboat

Before we move on: we did a lot of work to clean up this data, but we haven't saved anything to our Vertica database! Let's look at the modifications we've made to the :py:mod:`~verticapy.vDataFrame`.

.. ipython:: python

    print(titanic.current_relation())

Let see what's happening when we aggregate and turn on SQL generation.

.. ipython:: python

    vp.set_option("sql_on", True)
    titanic_avg = titanic.avg()

VerticaPy dynamically generates SQL code whenever you make modifications to your data. To avoid recomputation, it also stores previous aggregations. If we filter anything in our data, it will update the catalog with our modifications.

.. ipython:: python

    vp.set_option("sql_on", False)
    print(titanic.info())

Let's move on to modeling our data. Save the :py:mod:`~verticapy.vDataFrame` to your Vertica database.

.. ipython:: python
    :okwarning:

    from verticapy.sql import drop

    # Titanic Boat
    drop("titanic_boat", method = "view")
    titanic_boat = titanic.search(titanic["boat"] == 1).to_db("titanic_boat", relation_type = "view")

    # Titanic No Boat
    drop("titanic_no_boat", method = "view")
    titanic_no_boat = titanic.search(titanic["boat"] == 0).to_db("titanic_no_boat", relation_type = "view")

Machine Learning
-----------------

Passengers with a lifeboat
+++++++++++++++++++++++++++

First, let's look at the number of survivors.

.. code-block:: python

    titanic_boat["survived"].describe()

.. ipython:: python
    :suppress:
    :okwarning:

    res = titanic_boat["survived"].describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_with_boat.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_with_boat.html

We have nine deaths. Let's try to understand why these passengers died.

.. code-block:: python

    titanic_boat.search(titanic_boat["survived"] == 0).head(10)

.. ipython:: python
    :suppress:
    :okwarning:

    res = titanic_boat.search(titanic_boat["survived"] == 0).head(10)
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_with_boat_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_with_boat_2.html

Apart from a decent amount of these passengers being third-class passengers, it doesn't seem like there are any clear predictors here for their deaths. Making a model from this would be unhelpful.

Passengers without a lifeboat
++++++++++++++++++++++++++++++

Let's move on to passengers without a lifeboat.

.. code-block:: python

    titanic_no_boat["survived"].describe()

.. ipython:: python
    :suppress:
    :okwarning:

    res = titanic_no_boat["survived"].describe()
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_without_boat.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_without_boat.html

Only 20 survived. Let's find out why.

.. code-block:: python

    titanic_no_boat.search(titanic_boat["survived"] == 1).head(20)

.. ipython:: python
    :suppress:
    :okwarning:

    res = titanic_no_boat.search(titanic_boat["survived"] == 1).head(20)
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_without_boat_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_without_boat_2.html

Most survivors seem to be women. Let's build a model with this in mind.

One of our predictors is categorical: the passenger title. Some of these predictors are corrleated, so it'd be best to work with a non-linear classifier that can handle that. In this case, a random forest classifier seems to be perfect. Let's evaluate it with a ``cross-validation``.

.. code-block:: python

    from verticapy.machine_learning.vertica import RandomForestClassifier
    from verticapy.machine_learning.model_selection import cross_validate

    predictors = titanic.get_columns(exclude_columns = ["survived"])
    response = "survived"
    model = RandomForestClassifier(
        n_estimators = 40, 
        max_depth = 4,
    )
    cross_validate(model, titanic_no_boat, predictors, response)

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import RandomForestClassifier
    from verticapy.machine_learning.model_selection import cross_validate

    predictors = titanic.get_columns(exclude_columns = ["survived"])
    response = "survived"
    model = RandomForestClassifier(
        n_estimators = 40, 
        max_depth = 4,
    )
    res = cross_validate(model, titanic_no_boat, predictors, response)
    html_file = open("SPHINX_DIRECTORY/figures/examples_titanic_table_ml_cv.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_ml_cv.html

This dataset is pretty unbalanced so we'll use an ``AUC`` to evaluate it. Looking at our table, our model has an average ``AUC`` of more than 0.9, so our model is quite good.

We can now build a model with the entire dataset.

.. ipython:: python
    :okwarning:

    model.fit(titanic_no_boat, predictors, response)

Let's look at the importance of each feature.

.. code-block:: python

    model.features_importance()

.. ipython:: python
    :suppress:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = model.features_importance()
    fig.write_html("SPHINX_DIRECTORY/figures/examples_titanic_table_features.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/examples_titanic_table_features.html

As expected, the passenger's title is the most important predictors of survival.

Conclusion
-----------

We've solved our problem in a Pandas-like way, all without ever loading data into memory!