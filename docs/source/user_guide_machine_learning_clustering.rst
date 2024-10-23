.. _user_guide.machine_learning.clustering:

===========
Clustering
===========

Clustering algorithms are used to segment data or to find anomalies. Generally speaking, clustering algorithms are sensitive to unnormalized data, so it's important to properly prepare your data beforehand.

For example, if we consider the 'titanic' dataset, the features 'fare' and 'age' don't have values within the same interval; that is, 'fare' can be much higher than the 'age'. Applying a clustering algorithm to this kind of dataset would create misleading clusters.

To create a clustering model, we'll start by importing the ``k-means`` algorithm.

.. ipython:: python
    
    import verticapy as vp
    from verticapy.machine_learning.vertica import KMeans

Next, we'll create a model object.

.. ipython:: python
    
    model = KMeans(n_cluster = 3)

Let's use the iris dataset to fit our model.

.. ipython:: python
    
    from verticapy.datasets import load_iris

    iris = load_iris()

We can then fit the model with our data.

.. ipython:: python
    
    model.fit(iris, ["PetalLengthCm", "SepalLengthCm"])

.. code-block:: python

    model.plot()

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = model.plot()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_ml_plot_clustering_1.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_plot_clustering_1.html

While there aren't any real metrics for evaluating unsupervised models, metrics used during computation can help us to understand the quality of the model. For example, a ``k-means`` model with fewer clusters and when the ``k-means`` score, 'Between-Cluster SS / Total SS' is close to 1.

.. ipython:: python

    print(model.get_vertica_attributes("metrics")["metrics"][0])

You can add the prediction to your vDataFrame.    

.. code-block::

    model.predict(iris, name = "cluster")

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.predict(iris, name = "cluster")
    html_file = open("/project/data/VerticaPy/docs/figures/ug_ml_table_clustering_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_ml_table_clustering_1.html

This concludes this lesson on clustering models in VerticaPy. We'll look at time series models in the next lesson.