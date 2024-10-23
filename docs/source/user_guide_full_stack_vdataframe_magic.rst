.. _user_guide.full_stack.vdataframe_magic:

======================================
The 'Magic' Methods of the vDataFrame
======================================

VerticaPy 0.3.2 introduces the 'Magic' methods, which offer some additional flexilibility for mathematical operations in the :py:mod:`vDataFrame`. These methods let you handle many operations in a 'pandas-like' or Pythonic style.

.. code-block:: ipython

    from verticapy.datasets import load_titanic

    titanic = load_titanic()
    titanic.head(100)

.. ipython:: python
    :suppress:
    :okwarning:

    from verticapy.datasets import load_titanic
    titanic = load_titanic()
    res = titanic.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_1.html

Feature Engineering, 'pandas'-style
------------------------------------

You can create new features with in a 'pandas' style.

.. code-block:: ipython

    titanic["family_size"] = titanic["parch"] + titanic["sibsp"] + 1
    titanic[["sibsp", "parch", "family_size"]]

.. ipython:: python
    :suppress:
    :okwarning:

    titanic["family_size"] = titanic["parch"] + titanic["sibsp"] + 1
    res = titanic[["sibsp", "parch", "family_size"]]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_2.html

You can also create features from various mathematical functions.

.. code-block:: ipython

    import verticapy.sql.functions as fun

    titanic["ln_fare"] = fun.ln(titanic["fare"])
    titanic[["fare", "ln_fare"]]

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy.sql.functions as fun
    titanic["ln_fare"] = fun.ln(titanic["fare"])
    res = titanic[["fare", "ln_fare"]]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_3.html

.. code-block:: ipython

    titanic["x"] = 1 - fun.exp(-titanic["fare"])
    titanic[["fare", "x"]]

.. ipython:: python
    :suppress:
    :okwarning:

    titanic["x"] = 1 - fun.exp(-titanic["fare"])
    res = titanic[["fare", "x"]]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_4.html

Conditional Operators
----------------------

You can now filter your data with conditional operators like and ('&'), or ('|'), equals ('=='), not equals (!=), and more!

Equal Operator (==)
++++++++++++++++++++

.. code-block:: ipython

    # Identifies the passengers who came alone

    single_family = titanic[titanic["family_size"] == 1]
    single_family.head(100)

.. ipython:: python
    :suppress:
    :okwarning:

    single_family = titanic[titanic["family_size"] == 1]
    res = single_family.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_5.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_5.html

And Operator (&)
+++++++++++++++++

.. code-block:: ipython

    # Identifies the passengers who came alone and 
    # who are between 15 and 24 years old...
    # ...with comparison operators
    single_family[(titanic["age"] >= 15) & (titanic["age"] <= 24)]

.. ipython:: python
    :suppress:
    :okwarning:

    res = single_family[(titanic["age"] >= 15) & (titanic["age"] <= 24)]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_6.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_6.html

Between Operator (_between)
++++++++++++++++++++++++++++

.. code-block:: ipython

    # ...with the 'between' function
    single_family[titanic["age"]._between(15, 24)]

.. ipython:: python
    :suppress:
    :okwarning:

    res = single_family[titanic["age"]._between(15, 24)]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_7.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_7.html

In Operator (_in)
++++++++++++++++++

.. code-block:: ipython

    # Identifies the passengers who came alone in 1st and 3rd class...

    # ...with the 'in' method
    single_family[titanic["pclass"]._in(1, 3)]

.. ipython:: python
    :suppress:
    :okwarning:

    res = single_family[titanic["pclass"]._in(1, 3)]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_8.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_8.html

Not In Operator (_not_in)
++++++++++++++++++++++++++

.. code-block:: ipython

    # ...with the 'not_in' method
    single_family[titanic["pclass"]._not_in(2)]

.. ipython:: python
    :suppress:
    :okwarning:

    res = single_family[titanic["pclass"]._not_in(2)]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_9.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_9.html

Or Operator (|)
++++++++++++++++

.. code-block:: ipython

    # Or operator
    single_family[(titanic["pclass"] == 1) | (titanic["pclass"] == 3)]

.. ipython:: python
    :suppress:
    :okwarning:

    res = single_family[(titanic["pclass"] == 1) | (titanic["pclass"] == 3)]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_10.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_10.html

Not Equal Operator (!=)
++++++++++++++++++++++++

.. code-block:: ipython

    # ...with the not equal operator
    single_family[titanic["pclass"] != 2]

.. ipython:: python
    :suppress:
    :okwarning:

    res = single_family[titanic["pclass"] != 2]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_11.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_11.html

'Pythonic' Invokation of Vertica Functions
-------------------------------------------

You can easily apply Vertica functions to your :py:mod:`vDataFrame`. Here, we use Vertica's COALESCE function to impute the 'age' of the passengers in our dataset.

.. code-block:: ipython

    titanic["age"].count()

.. ipython:: python

    res = titanic["age"].count()

.. ipython:: python

    titanic["age"] = fun.coalesce(titanic["age"], titanic["age"].avg());
    titanic["age"].count()

Slicing the vDataFrame
-----------------------

You can now slice the :py:mod:`vDataFrame` with indexing operators.

.. code-block:: ipython

    titanic[0:30]

.. ipython:: python
    :suppress:
    :okwarning:

    res = titanic[0:30]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_14.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_14.html

To access a single row, we just have to specify the index.

.. code-block:: python

    titanic[0]

.. ipython:: python
    :suppress:
    :okwarning:

    res = titanic[0]
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_15.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_table_vdfm_15.html
