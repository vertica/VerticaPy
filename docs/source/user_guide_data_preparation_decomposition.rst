.. _user_guide.data_preparation.decomposition:

==============
Decomposition
==============

Decomposition is the process of using an orthogonal transformation to convert a set of observations of possibly-correlated variables (with numerical values) into a set of values of linearly-uncorrelated variables called principal components.

Since some algorithms are sensitive to correlated predictors, it can be a good idea to use the :py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA` (Principal Component Analysis: Decomposition Technique) before applying the algorithm. Since some algorithms are also sensitive to the number of predictors, we'll have to be picky with which variables we include.

To demonstrate data decomposition in VerticaPy, we'll use the well-known ``iris`` dataset.

.. code-block:: python

    from verticapy.datasets import load_iris

    iris = load_iris()
    iris.head(100)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_iris
    iris = load_iris()
    res = iris.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_decomposition_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_decomposition_1.html

Notice that all the predictors are well-correlated with each other.

.. code-block:: python

    iris.corr()

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = iris.corr()
    fig.write_html("SPHINX_DIRECTORY/figures/ug_dp_plot_decomposition_2.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_plot_decomposition_2.html

Let's compute the :py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA` of the different elements.

.. ipython:: python

    from verticapy.machine_learning.vertica import PCA

    model = PCA()
    model.fit(
        iris, 
        [
            "PetalLengthCm", 
            "SepalWidthCm",
            "SepalLengthCm",
            "PetalWidthCm",
        ],
    )

Let's compute the correlation matrix of the result of the :py:mod:`~verticapy.machine_learning.vertica.decomposition.PCA`.

.. code-block:: python

    model.transform().corr()

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = model.transform().corr()
    fig.write_html("SPHINX_DIRECTORY/figures/ug_dp_plot_decomposition_3.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_plot_decomposition_3.html

Notice that the predictors are now independant and combined together and they have the exact same amount of information than the previous variables. Let's look at the accumulated explained variance of the PCA components.

.. ipython:: python

    model.explained_variance_

Most of the information is in the first two components with more than 97.7% of explained variance. We can export this result to a :py:mod:`~verticapy.vDataFrame`.

.. code-block::

    model.transform(n_components = 2)

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.transform(n_components = 2)
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_decomposition_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_decomposition_4.html