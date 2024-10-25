.. _user_guide.data_preparation.normalization:

==============
Normalization
==============

Normalizing data is crucial when using machine learning algorithms because of how sensitive most of them are to un-normalized data. For example, the neighbors-based and ``k-means`` algorithms use the ``p-distance`` in their learning phase. Normalization is the first step before using a linear regression due to Gauss-Markov assumptions.

Unnormalized data can also create complications for the convergence of some ML algorithms. Normalization is also a way to encode the data and to retain the global distribution. When we know the estimators to use to normalize the data, we can easily un-normalize the data and come back to the original distribution.

There are three main normalization techniques:

- **Z-Score:** We reduce and center the feature values using the average and standard deviation. This normalization is sensitive to outliers.
- **Robust Z-Score:** We reduce and center the feature values using the median and the median absolute deviation. This normalization is robust to outliers.
- **Min-Max:**  We reduce the feature values by using a bijection to ``[0,1]``. The max will reach 1 and the min will reach 0. This normalization is robust to outliers.

To demonstrate data normalization in VerticaPy, we will use the well-known 'Titanic' dataset.

.. code-block:: python

    from verticapy.datasets import load_titanic

    titanic = load_titanic()
    titanic.head(100)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_titanic
    titanic = load_titanic()
    res = titanic.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_norm_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_norm_1.html

Let's look at the 'fare' and 'age' of the passengers.

.. code-block:: python

    titanic.select(["age", "fare"])

.. ipython:: python
    :suppress:

    res = titanic.select(["age", "fare"])
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_norm_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_norm_2.html

These lie in different numerical intervals so it's probably a good idea to normalize them. To normalize data in VerticaPy, we can use the :py:func:`~verticapy.vDataFrame.normalize` method.

.. ipython:: python

    help(titanic["age"].normalize)

The three main normalization techniques are available. Let's normalize the 'fare' and the 'age' using the 'MinMax' method.

.. code-block:: python

    titanic["age"].normalize(method = "minmax")
    titanic["fare"].normalize(method = "minmax")
    titanic.select(["age", "fare"])

.. ipython:: python
    :suppress:

    titanic["age"].normalize(method = "minmax")
    titanic["fare"].normalize(method = "minmax")
    res = titanic.select(["age", "fare"])
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_norm_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_norm_3.html

Both of the features now scale in ``[0,1]``. It is also possible to normalize by a specific partition with the ``by`` parameter.