.. _user_guide.data_preparation.joins:

======
Joins
======

When working with datasets, we often need to merge data from different sources. To do this, we need keys on which to join our data.

Let's use the `US Flights 2015 datasets <https://www.kaggle.com/datasets/usdot/flight-delays>`_. We have three datasets.

First, we have information on each flight.

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

These lie in different numerical intervals so it's probably a good idea to normalize them. To normalize data in VerticaPy, we can use the ``normalize`` method.

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