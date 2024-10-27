.. _user_guide.full_stack.train_test_split:

=================
Train Test Split
=================

Before you test a supervised model, you'll need separate, non-overlapping sets for training and testing.

In VerticaPy, the :py:func:`~verticapy.vDataFrame.train_test_split` method uses a random number generator to decide how to split the data.

.. code-block:: ipython

    %load_ext verticapy.sql

.. code-block:: ipython
    
    %sql -c "SELECT SEEDED_RANDOM(0);"

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    res = vp.vDataFrame("SELECT SEEDED_RANDOM(0);")
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_tts_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_tts_1.html

The ``SEEDED_RANDOM`` function chooses a number in the interval ``[0,1)``. Since the seed is user-provided, these results are reproducible. In this example, passing ``0`` as the seed always returns the same value.

.. code-block:: ipython
    
    %sql -c "SELECT SEEDED_RANDOM(0);"

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    res = vp.vDataFrame("SELECT SEEDED_RANDOM(0);")
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_tts_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_tts_2.html

A different seed will generate a different value.

.. code-block:: ipython
    
    %%sql -c "SELECT SEEDED_RANDOM(1);"

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    res = vp.vDataFrame("SELECT SEEDED_RANDOM(1);")
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_tts_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_tts_3.html

The :py:func:`~verticapy.vDataFrame.train_test_split` function generates a random seed and we can then share that seed between the training and testing sets.

.. ipython:: python

    from verticapy.datasets import load_titanic

    titanic = load_titanic()
    train, test = titanic.train_test_split()

.. ipython:: python

    titanic.shape()

.. ipython:: python

    train.shape()

.. ipython:: python

    test.shape()

Note that ``SEEDED_RANDOM`` depends on the order of your data. That is, if your data isn't sorted by a unique feature, the selected data might be inconsistent. To avoid this, we'll want to use the ``order_by`` parameter.

.. ipython:: python

    train, test = titanic.train_test_split(order_by = {"fare": "asc"})

Even if the ``fare`` has duplicates, ordering the data alone will drastically decrease the likelihood of a collision.

Let's create a model and evaluate it.

.. ipython:: python

    from verticapy.machine_learning.vertica import LinearRegression

    model = LinearRegression()

When fitting the model with the :py:func:`~verticapy.machine_learning.vertica.LinearRegression.fit` method, you can use the parameter ``test_relation`` to score your data on a specific relation.

.. ipython:: python

    model.fit(
        train,
        ["age", "fare"],
        "survived",
        test,
    )

.. code-block:: ipython
    
    model.report()

.. ipython:: python
    :suppress:
    :okwarning:

    res = model.report()
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_tts_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_tts_4.html

All model evaluation abstractions will now use the test relation for the scoring. After that, you can evaluate the efficiency of your model.