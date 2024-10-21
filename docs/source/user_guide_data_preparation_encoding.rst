.. _user_guide.data_preparation.encoding:

=========
Encoding
=========

Encoding features is a very important part of the data science life cycle. In data science, generality is important and having too many categories can compromise that and lead to incorrect results. In addition, some algorithmic optimizations are linear and prefer categorized information, and some can't process non-numerical features.

There are many encoding techniques:

- **User-Defined Encoding:** The most flexible encoding. The user can choose how to encode the different categories.
- **Label Encoding:** Each category is converted to an integer using a bijection to [0;n-1] where n is the feature number of unique values.
- **One-hot Encoding:** This technique creates dummies (values in {0,1}) of each category. The categories are then separated into n features.
- **Mean Encoding:** This technique uses the frequencies of each category for a specific response column.
- **Discretization:** This technique uses various mathematical technique to encode continuous features into categories.

To demonstrate encoding data in VerticaPy, we'll use the well-known 'Titanic' dataset.

.. code-block:: python

    from verticapy.datasets import load_titanic

    titanic = load_titanic()
    titanic.head(100)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_titanic
    titanic = load_titanic()
    res = titanic.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_encoding_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_encoding_1.html

Let's look at the 'age' of the passengers.

.. code-block:: python

    titanic["age"].hist()

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = titanic["age"].hist()
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_encoding_2.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_encoding_2.html

By using the ``discretize`` method, we can discretize the data using equal-width binning.

.. code-block:: python

    titanic["age"].discretize(method = "same_width", h = 10)
    titanic["age"].bar(max_cardinality = 10)

.. ipython:: python
    :suppress:
    :okwarning:

    titanic["age"].discretize(method = "same_width", h = 10)
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = titanic["age"].bar(max_cardinality = 10)
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_encoding_3.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_encoding_3.html

We can also discretize the data using frequency bins.

.. code-block:: python

    titanic = load_titanic()
    titanic["age"].discretize(method = "same_freq", nbins = 5)
    titanic["age"].bar(max_cardinality = 5)

.. ipython:: python
    :suppress:
    :okwarning:

    titanic = load_titanic()
    titanic["age"].discretize(method = "same_freq", nbins = 5)
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = titanic["age"].bar(max_cardinality = 5)
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_encoding_4.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_encoding_4.html

Computing categories using a response column can also be a good solution.

.. code-block:: python

    titanic = load_titanic()
    titanic["age"].discretize(method = "smart", response = "survived", nbins = 6)
    titanic["age"].bar(method = "avg", of = "survived")

.. ipython:: python
    :suppress:
    :okwarning:

    titanic = load_titanic()
    titanic["age"].discretize(method = "smart", response = "survived", nbins = 6)
    import verticapy
    verticapy.set_option("plotting_lib", "plotly")
    fig = titanic["age"].bar(method = "avg", of = "survived")
    fig.write_html("/project/data/VerticaPy/docs/figures/ug_dp_plot_encoding_5.html")

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_plot_encoding_5.html

We can view the available techniques in the ``discretize`` method with the ``help`` method.

.. ipython:: python

    help(titanic["age"].discretize)

To encode a categorical feature, we can use label encoding. For example, the column 'sex' has two categories (male and female) that we can represent with 0 and 1, respectively.

.. code-block:: python

    titanic["sex"].label_encode()
    titanic["sex"].head(100)

.. ipython:: python
    :suppress:

    titanic["sex"].label_encode()
    res = titanic["sex"].head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_encoding_6.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_encoding_6.html

When a feature has few categories, the most suitable choice is the one-hot encoding. Label encoding converts a categorical feature to numerical without retaining its mathematical relationships. Let's use a one-hot encoding on the 'embarked' column.

.. code-block:: python

    titanic["embarked"].one_hot_encode()
    titanic.select(["embarked", "embarked_C", "embarked_Q"])

.. ipython:: python
    :suppress:

    titanic["embarked"].one_hot_encode()
    res = titanic.select(["embarked", "embarked_C", "embarked_Q"])
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_encoding_7.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_encoding_7.html

One-hot encoding can be expensive if the column in question has a large number of categories. In that case, we should use mean encoding. Mean encoding replaces each category of a variable with its corresponding average over a partition by a response column. This makes it an efficient way to encode the data, but be careful of over-fitting.

Let's use a mean encoding on the 'home.dest' variable.

.. code-block:: python

    titanic["home.dest"].mean_encode("survived")
    titanic.head(100)

.. ipython:: python
    :suppress:

    titanic["home.dest"].mean_encode("survived")
    res = titanic.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_encoding_8.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_encoding_8.html

VerticaPy offers many encoding techniques. For example, the ``case_when`` and ``decode`` methods allow the user to use a customized encoding on a column. The ``discretize`` method allows you to reduce the number of categories in a column. It's important to get familiar with all the techniques available so you can make informed decisions about which to use for a given dataset.