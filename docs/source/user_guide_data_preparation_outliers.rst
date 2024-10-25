.. _user_guide.data_preparation.outliers:

=========
Outliers
=========

Outliers are data points that differ significantly from the rest of the data. While some outliers can reveal some important information (machine failure, systems fraud...), they can also be simple errors.

Some machine learning algorithms are sensitive to outliers. In fact, they can destroy the final predictions because of how much bias they add to the data, and handling outliers in our data is one of the most important parts of the data preparation.

Outliers consist of three main types:

- **Global Outliers:** Values far outside the entirety of their source dataset.
- **Contextual Outliers:** Values deviate significantly from the rest of the data points in the same context.
- **Collective Outliers:** Values that aren't global or contextual outliers, but as a collection deviate significantly from the entire dataset.

Global outliers are often the most critical type and can add a significant amount of bias into the data. Fortunately, we can easily identify these outliers by computing the ``Z-Score``.

Let's look at some examples using the `Heart Disease <https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset>`_ dataset. This dataset contains information on patients who are likely to have heart-related complications.

.. code-block:: python

    import verticapy as vp

    heart = vp.read_csv("heart.csv")
    heart.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    heart = vp.read_csv("/project/data/VerticaPy/docs/source/_static/website/examples/data/heart/heart.csv")
    res = heart.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_outliers_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_outliers_1.html

Let's focus on a patient's maximum heart rate (thalach) and the cholesterol (chol) to identify some outliers.

.. code-block:: python

    heart.scatter(["thalach", "chol"])

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = heart.scatter(["thalach", "chol"])
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_2.html

We can see some outliers of the distribution: people with high cholesterol and others with a very low heart rate. Let's compute the global outliers using the :py:func:`~verticapy.vDataFrame.outlier` method.

.. code-block:: python

    heart.outliers(["thalach", "chol"], "global_outliers")
    heart.scatter(["thalach", "chol"], by = "global_outliers")

.. ipython:: python
    :suppress:
    :okwarning:

    heart.outliers(["thalach", "chol"], "global_outliers")
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = heart.scatter(["thalach", "chol"], by = "global_outliers")
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_3.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_3.html

It is also possible to draw an outlier plot using the :py:func:`~verticapy.vDataFrame.outliers_plot` method.

.. code-block:: python

    heart.outliers_plot(["thalach", "chol"],)

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = heart.outliers_plot(["thalach", "chol"],)
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_4.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_4.html

We've detected some global outliers in the distribution and we can impute these with the :py:func:`~verticapy.vDataFrame.fill_outliers` method.

Generally, you can identify global outliers with the ``Z-Score``. We'll consider a ``Z-Score`` greater than 3 indicates that the datapoint is an outlier. Some less precise techniques consider the data points belonging in the first and last alpha-quantile as outliers. You're free to choose either of these strategies when filling outliers.

.. code-block:: python

    heart["thalach"].fill_outliers(
        use_threshold = True,
        threshold = 3.0,
        method = "winsorize",
    )
    heart["chol"].fill_outliers(
        use_threshold = True,
        threshold = 3.0,
        method = "winsorize",
    )
    heart.scatter(
        ["thalach", "chol"],
        by = "global_outliers",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    heart["thalach"].fill_outliers(
        use_threshold = True,
        threshold = 3.0,
        method = "winsorize",
    )
    heart["chol"].fill_outliers(
        use_threshold = True,
        threshold = 3.0,
        method = "winsorize",
    )
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = heart.scatter(
        ["thalach", "chol"],
        by = "global_outliers",
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_5.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_5.html

Other techniques like :py:mod:`~verticapy.machine_learning.vertica.DBSCAN` or local outlier factor (``LOF``) can be to used to check other data points for outliers.

.. code-block:: python

    from verticapy.machine_learning.vertica import DBSCAN

    model = DBSCAN(eps = 20, min_samples = 10)
    model.fit(heart, ["thalach", "chol"])
    model.plot()

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import DBSCAN

    model = DBSCAN(eps = 20, min_samples = 10)
    model.fit(heart, ["thalach", "chol"])
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = model.plot()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_6.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_6.html

.. code-block:: python

    heart_dbscan = model.predict()
    heart_dbscan["outliers_dbscan"] = "(dbscan_cluster = -1)::int"
    heart_dbscan.scatter(
        ["thalach", "chol"],
        by = "outliers_dbscan",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    heart_dbscan = model.predict()
    heart_dbscan["outliers_dbscan"] = "(dbscan_cluster = -1)::int"
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = heart_dbscan.scatter(
        ["thalach", "chol"],
        by = "outliers_dbscan",
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_7.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_7.html

While :py:mod:`~verticapy.machine_learning.vertica.DBSCAN` identifies outliers when computing the clusters, ``LOF`` computes an outlier score. Generally, a ``LOF`` Score greater than 1.5 indicates an outlier.

.. code-block:: python

    from verticapy.machine_learning.vertica import LocalOutlierFactor

    model = LocalOutlierFactor()
    model.fit(heart, ["thalach", "chol",])
    model.plot()

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.machine_learning.vertica import LocalOutlierFactor

    model = LocalOutlierFactor()
    model.fit(heart, ["thalach", "chol",])
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = model.plot()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_8.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_8.html

.. code-block:: python

    heart_lof = model.predict()
    heart_lof["outliers"] = "(CASE WHEN lof_score > 1.5 THEN 1 ELSE 0 END)"
    heart_lof.scatter(
        ["thalach", "chol"],
        by = "outliers",
    )

.. ipython:: python
    :suppress:
    :okwarning:

    heart_lof = model.predict()
    heart_lof["outliers"] = "(CASE WHEN lof_score > 1.5 THEN 1 ELSE 0 END)"
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = heart_lof.scatter(
        ["thalach", "chol"],
        by = "outliers",
    )
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_9.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_outliers_9.html

We have many other techniques like the ``k-means`` clustering for finding outliers, but the most important method is using the ``Z-Score``. After identifying outliers, we just have to decide how to impute the missing values. We'll focus on missing values in the next lesson.