.. _user_guide.data_preparation.features_engineering:

=====================
Features Engineering
=====================

While preparing our data, we need to think constantly about the most suitable features we can use to achieve our overall goals.
Features engineering makes use of many techniques - too many to go over in this short lesson. We'll focus on the most popular ones.

Customized Features Engineering
--------------------------------

To build a customized feature, you can use the :py:func:`~verticapy.vDataFrame.eval` method of the :py:func:`~verticapy.vDataFrame`. Let's look at an example with the well-known 'Titanic' dataset.

.. code-block:: python
    
    import verticapy as vp
    from verticapy.datasets import load_titanic

    titanic = load_titanic()
    titanic.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    from verticapy.datasets import load_titanic
    titanic = load_titanic()
    res = titanic.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_fe_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_fe_1.html

The feature 'parch' corresponds to the number of parents and children on-board. The feature 'sibsp' corresponds to the number of siblings and spouses on-board. We can create the feature 'family size' which is equal to parch + sibsp + 1.

.. code-block:: python
    
    titanic["family_size"] = titanic["parch"] + titanic["sibsp"] + 1
    titanic.select(["parch", "sibsp", "family_size"])

.. ipython:: python
    :suppress:

    titanic["family_size"] = titanic["parch"] + titanic["sibsp"] + 1
    res = titanic.select(["parch", "sibsp", "family_size"])
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_fe_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_fe_2.html

When using the :py:func:`~verticapy.vDataFrame.eval` method, you can enter any SQL expression and VerticaPy will evaluate it!

Regular Expressions
--------------------

To compute features using regular expressions, we'll use the :py:func:`~verticapy.vDataFrame.regexp` method.

.. ipython:: python

    help(vp.vDataFrame.regexp)

Consider the following example: notice that passenger names include their title.

.. code-block:: python
    
    titanic["name"]

.. ipython:: python
    :suppress:

    res = titanic["name"]
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_fe_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_fe_3.html

Let's extract the title using regular expressions.

.. code-block:: python
    
    titanic.regexp(
        column = "name",
        name = "title",
        pattern = " ([A-Za-z])+\.",
        method = "substr",
    )
    titanic.select(["name", "title"])

.. ipython:: python
    :suppress:

    titanic.regexp(
        column = "name",
        name = "title",
        pattern = " ([A-Za-z])+\.",
        method = "substr",
    )
    res = titanic.select(["name", "title"])
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_fe_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_fe_4.html

Advanced Analytical Functions
------------------------------

The 'analytic' method contains the many advanced analytical functions in VerticaPy.

.. ipython:: python

    help(vp.vDataFrame.analytic)

To demonstrate some of these techniques, let's use the Amazon dataset and perform some computations.

.. code-block:: python
    
    from verticapy.datasets import load_amazon

    amazon = load_amazon()
    amazon.head(100)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_amazon
    amazon = load_amazon()
    res = amazon.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_fe_5.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_fe_5.html

For each state, let's compute the previous number of forest fires.

.. code-block:: python
    
    amazon.analytic(
        name = "previous_number",
        func = "lag",
        columns = "number",
        by = "state",
        order_by = {"date": "asc"},
    )

.. ipython:: python
    :suppress:

    res = amazon.analytic(
        name = "previous_number",
        func = "lag",
        columns = "number",
        by = "state",
        order_by = {"date": "asc"},
    )
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_fe_6.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_fe_6.html

Moving Windows
---------------

Moving windows are powerful features. Moving windows are managed by the :py:func:`~verticapy.vDataFrame.rolling` method in VerticaPy.

.. ipython:: python

    help(vp.vDataFrame.rolling)

Let's look at forest fires for each state three months preceding two months following the examined period.

.. code-block:: python
    
    amazon.rolling(
        name = "number_3mp_2mf",
        func = "sum",
        window = ("- 3 months", "2 months"),
        columns = "number",
        by = "state",
        order_by = {"date": "asc"},
    )

.. ipython:: python
    :suppress:

    res = amazon.rolling(
        name = "number_3mp_2mf",
        func = "sum",
        window = ("- 3 months", "2 months"),
        columns = "number",
        by = "state",
        order_by = {"date": "asc"},
    )
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_fe_7.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_fe_7.html

Moving windows give us infinite possibilities for creating new features.

After we've finished preparing our data, our next task is to create a machine learning model.