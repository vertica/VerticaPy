.. _user_guide.data_preparation.duplicates:

===========
Duplicates
===========

When merging different data sources, we're likely to end up with duplicates that can add a lot of bias to and skew our data. Just imagine running a Telco marketing campaign and not removing your duplicates: you'll end up targeting the same person multiple times!

Let's use the ``iris`` dataset to understand the tools VerticaPy gives you for handling duplicate values.

.. code-block:: python

    from verticapy.datasets import load_iris

    iris = load_iris()
    iris = iris.append(load_iris().sample(3)) # adding some duplicates
    iris.head(100)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_iris
    iris = load_iris()
    iris = iris.append(load_iris().sample(3)) # adding some duplicates
    res = iris.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_duplicates_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_duplicates_1.html

To find all the duplicates, you can use the :py:func:`~verticapy.vDataFrame.duplicated` method.

.. code-block:: python

    iris.duplicated()

.. ipython:: python
    :suppress:

    res = iris.duplicated()
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_duplicates_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_duplicates_2.html

As you might expect, some flowers might share the exact same characteristics. But we have to be careful; this doesn't mean that they are real duplicates. In this case, we don't have to drop them.

That said, if we did want to drop these duplicates, we can do so with the :py:func:`~verticapy.vDataFrame.drop_duplicates` method.

.. code-block:: python

    iris.drop_duplicates()

.. ipython:: python
    :suppress:

    res = iris.drop_duplicates()
    html_file = open("SPHINX_DIRECTORY/figures/ug_dp_table_duplicates_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_dp_table_duplicates_3.html

Using this method will add an advanced analytical function to the SQL code generation which is quite expensive. You should only use this method after aggregating the data to avoid stacking heavy computations on top of each other.