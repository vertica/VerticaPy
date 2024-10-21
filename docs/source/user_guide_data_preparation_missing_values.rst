.. _user_guide.data_preparation.missing_values:

===============
Missing Values
===============

Missing values occur when no data value is stored for the variable in an observation and are most often represented with a NULL or None. Not handling them can lead to unexpected results (for example, some ML algorithms can't handle missing values at all) and worse, it can lead to incorrect conclusions.

There are 3 main types of missing values:

- **MCAR (Missing Completely at Random):** The events that lead to any particular data-item being missing occur entirely at random. For example, in IOT, we can lose sensory data in transmission.
- **MAR (Missing {Conditionally} at Random):** Missing data doesn't happen at random and is instead related to some of the observed data. For example, some students may have not answered to some specific questions of a test because they were absent during the relevant lesson.
- **MNAR (Missing not at Random):** The value of the variable that’s missing is related to the reason it’s missing. For example, if someone didn’t subscribe to a loyalty program, we can leave the cell empty.

Different types of missing values tend to suggest different methods for imputing them. For example, when dealing with MCAR values, you can use mathematical aggregations to impute the missing values. For MNAR values, we can simply create another category. MAR values, however, we'll need to do some more investigation before deciding how to impute the data.

To see how to handle missing values in VerticaPy, we'll use the well-known 'Titanic' dataset.

.. code-block:: python

    from verticapy.datasets import load_titanic

    titanic = load_titanic()
    titanic.head(100)

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_titanic
    titanic = load_titanic()
    res = titanic.head(100)
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_mv_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_mv_1.html

We can examine the missing values with the ``count`` method.

.. code-block:: python

    titanic.count_percent()

.. ipython:: python
    :suppress:

    res = titanic.count_percent()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_mv_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_mv_2.html

The missing values for 'boat' are MNAR; missing values simply indicate that the passengers didn't pay for a lifeboat. We can replace all the missing values with a new category 'No Lifeboat' using the ``fillna`` method.

.. code-block:: python

    titanic["boat"].fillna("No Lifeboat")
    titanic["boat"]

.. ipython:: python
    :suppress:

    titanic["boat"].fillna("No Lifeboat")
    res = titanic["boat"]
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_mv_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_mv_3.html

Missing values for 'age' seem to be MCAR, so the best way to impute them is with mathematical aggregations. Let's impute the age using the average age of passengers of the same sex and class.

.. code-block:: python

    titanic["age"].fillna(
        method = "avg",
        by = ["pclass", "sex"],
    )
    titanic["age"]

.. ipython:: python
    :suppress:

    titanic["age"].fillna(
        method = "avg",
        by = ["pclass", "sex"],
    )
    res = titanic["age"]
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_mv_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_mv_4.html

The features 'embarked' and 'fare' have a couple missing values. Instead of using a technique to impute them, we can just drop them with the ``dropna`` method.

.. code-block:: python

    titanic["fare"].dropna()
    titanic["embarked"].dropna()

.. ipython:: python
    :suppress:

    titanic["fare"].dropna()
    res = titanic["embarked"].dropna()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_dp_table_mv_5.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_dp_table_mv_5.html

The ``fillna`` method offers many options. Let's use the ``help`` method to view its parameters.

.. ipython:: python

    help(titanic["embarked"].fillna)

.. ipython:: python
    
    print(titanic.current_relation())

Depending on the circumstances, we'll need to investigate to find the most suitable solution.

In conclusion, before imputing missing data, you have to understand why it might be missing and how it relates to the rest of your dataset.