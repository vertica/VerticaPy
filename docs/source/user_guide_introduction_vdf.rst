.. _user_guide.introduction.vdf:

The Virtual DataFrame
=====================

The Virtual DataFrame (vDataFrame) is the core object of the VerticaPy library. Leveraging the power of Vertica and the flexibility of Python, the :py:func:`~verticapy.vDataFrame` is a Python object that lets you manipulate the data representation in a Vertica database without modifying the underlying data. The data represented by a :py:func:`~verticapy.vDataFrame` remains in the Vertica database, bypassing the limitations of working memory. When a :py:func:`~verticapy.vDataFrame` is created or altered, VerticaPy formulates the operation as an SQL query and pushes the computation to the Vertica database, harnessing Vertica's massive parallel processing and in-built functions. Vertica then aggregates and returns the result to VerticaPy. In essence, vDataFrames behave similar to `views <https://docs.vertica.com/latest/en/data-analysis/views/>`_ in the Vertica database.

For more information about Vertica's performance advantages, including its columnar orientation and parallelization across 
nodes, see the `Vertica documentation <https://docs.vertica.com/latest/en/architecture/>`_.

In the following tutorial, we will introduce the basic functionality of the :py:func:`~verticapy.vDataFrame` and then explore the ways in which they utilize in-database processing to enhance performance. 

Creating vDataFrames
---------------------

First, run the :py:func:`~verticapy.datasets.load_titanic` function to ingest into 
Vertica a dataset with information about titanic passengers:

.. code-block:: python

    from verticapy.datasets import load_titanic

    load_titanic()

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_titanic
    res = load_titanic()
    html_file = open("/project/data/VerticaPy/docs/figures/user_guide_introduction_best_practices_laod_titanic.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/user_guide_introduction_best_practices_laod_titanic.html

You can create a :py:func:`~verticapy.vDataFrame` from either an existing relation or a customized relation.

To create a :py:func:`~verticapy.vDataFrame` using an existing relation, in this case the Titanic dataset, provide the name of the dataset:

.. code-block:: python

    import verticapy as vp

    vp.vDataFrame("public.titanic")

To create a :py:func:`~verticapy.vDataFrame` using a customized relation, specify the SQL query for that relation as the argument:

.. code-block:: python

    vp.vDataFrame("SELECT pclass, AVG(survived) AS survived FROM titanic GROUP BY 1")

.. ipython:: python
    :suppress:

    import verticapy as vp
    res = vp.vDataFrame("SELECT pclass, AVG(survived) AS survived FROM titanic GROUP BY 1")
    html_file = open("/project/data/VerticaPy/docs/figures/ug_intro_vdf_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_intro_vdf_1.html

For more examples of creating vDataFrames, see :py:func:`~verticapy.vDataFrame`.

In-memory vs. in-database
--------------------------

The following examples demonstrate the performance advantages of loading and processing data in-database versus in-memory.

First, we download the `Expedia dataset <https://www.kaggle.com/competitions/expedia-hotel-recommendations/data>`_ from Kaggle and then load it into Vertica:

.. note:: 
    
    In this example, we are only taking a subset of the entire dataset to save time. 
    But as per our studies, the benifits increase exponentially if the size of the data gets larger.

.. code-block:: python

    vp.read_csv("expedia.csv", schema = "public", parse_nrows = 20000000)

Once the data is loaded into the Vertica database, we can create a :py:func:`~verticapy.vDataFrame` using the relation that contains the Expedia dataset:

.. ipython:: python

    import time
    @suppress
    vp.drop("public.expedia")
    @suppress
    vp.read_csv(
        "/project/data/VerticaPy/docs/source/_static/website/examples/data/booking/expedia.csv",
        schema = "public", 
        parse_nrows = 20000000,
    )
    start_time = time.time()
    expedia = vp.vDataFrame("public.expedia")
    print("elapsed time = {}".format(time.time() - start_time))

The :py:func:`~verticapy.vDataFrame` was created in about a second. All the data—about 4GB—is stored in Vertica, requiring no in-memory data loading.

Now, to compare the above result with in-memory loading, we load about half the dataset into pandas:

.. note::

     This process is expensive on local machines, so 
     avoid running the following code if your computer 
     has less than 2GB of memory.

.. code-block:: python

    import pandas as pd

    L_nrows = [10000, 100000, 149814]
    L_time = []
    for nrows in L_nrows:
        start_time = time.time()
        expedia_df = pd.read_csv("expedia.csv", nrows = nrows)
        elapsed_time = time.time() - start_time
        L_time.append(elapsed_time)

.. code-block:: python
    
    import pandas as pd

    L_nrows = [10000, 100000, 149814]
    L_time = []
    for nrows in L_nrows:
        start_time = time.time()
        expedia_df = pd.read_csv(
            "expedia.csv",
            nrows = nrows,
        )
        elapsed_time = time.time() - start_time
        L_time.append(elapsed_time)

.. ipython:: python
    :suppress:

    import pandas as pd

    L_nrows = [10000, 100000, 149814]
    L_time = []
    for nrows in L_nrows:
        start_time = time.time()
        expedia_df = pd.read_csv(
            "/project/data/VerticaPy/docs/source/_static/website/examples/data/booking/expedia.csv",
            nrows = nrows,
        )
        elapsed_time = time.time() - start_time
        L_time.append(elapsed_time)

.. ipython:: python

    for i in range(len(L_time)):
        print("nrows = {}; elapsed time = {}".format(L_nrows[i], L_time[i]))

It took an order of magnitude more to load into memory compared with the time required to create the :py:func:`~verticapy.vDataFrame`. Loading data into pandas is quite fast when the data volume is low (less than some MB), but as the size of the dataset increases, the load time can become exponentially more expensive, as seen in the following plot:

.. ipython:: python

    import matplotlib.pyplot as plt
    @savefig ug_intro_vdf_plot
    plt.plot(L_nrows, L_time)
    @savefig ug_intro_vdf_plot_2
    plt.show()

Even after the data is loaded into memory, the performance is very slow. The following example removes non-numeric columns from the dataset, then computes a correlation matrix:

.. ipython:: python

    columns_to_drop = ['date_time', 'srch_ci', 'srch_co'] ;
    expedia_df = expedia_df.drop(columns_to_drop, axis=1);
    start_time = time.time()
    expedia_df.corr();
    print(f"elapsed time = {time.time() - start_time}")

Let's compare the performance in-database using a :py:func:`~verticapy.vDataFrame` to compute the correlation matrix of the entire dataset:

.. ipython:: python

    # Remove non-numeric columns
    expedia.drop(columns = ['date_time', 'srch_ci', 'srch_co']);
    start_time = time.time()
    expedia.corr(show = False);
    print(f"elapsed time = {time.time() - start_time}")

VerticaPy also caches the computed aggregations. With this cache available, we can repeat the correlation matrix computation almost instantaneously:

.. note:: 
    
    If necessary, you can deactivate the cache by calling the :py:func:`~verticapy.set_option` function with the `cache` parameter set to False.

.. ipython:: python

    start_time = time.time()
    expedia.corr(show = False);
    print(f"elapsed time = {time.time() - start_time}")

Memory usage 
+++++++++++++

Now, we will examine how the memory usage compares between in-memory and in-database.

First, use the pandas `info()` method to explore the DataFrame's memory usage:

.. ipython:: python

    expedia_df.info()

Compare this with vDataFrame:

.. code-block:: python

    expedia.memory_usage()

.. ipython:: python
    :suppress:

    res = expedia.memory_usage()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_intro_vdf_mem.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_intro_vdf_mem.html

The :py:func:`~verticapy.vDataFrame` only uses about 37KB! By storing the data in the Vertica database, and only recording the 
user's data modifications in memory, the memory usage is reduced to a minimum. 

With VerticaPy, we can take advantage of Vertica's structure and scalability, providing fast queries without ever loading the data into memory. In the above examples, we've seen that in-memory processing is much more expensive in both computation and memory usage. This often leads to the decesion to downsample the data, which sacrfices the possibility of further data insights.

The :py:func:`~verticapy.vDataFrame` structure
----------------------------------------------

Now that we've seen the performance and memory benefits of the vDataFrame, let's dig into some of the underlying structures and methods that produce these great results.

:py:func:`~verticapy.vDataFrame`s are composed of columns called :py:mod:`vDataColumn`s. To view all :py:mod:`vDataColumn`s in a :py:func:`~verticapy.vDataFrame`, use the :py:func:`~verticapy.get_columns` method:

.. ipython:: python

    expedia.get_columns()

To access a :py:func:`~verticapy.vDataColumn`, specify the column name in square brackets, for example:

.. note::

    VerticaPy saves computed aggregations to avoid unncessary recomputations.

.. code-block:: python

    expedia["is_booking"].describe()

.. ipython:: python
    :suppress:

    res = expedia["is_booking"].describe()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_intro_vdf_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_intro_vdf_describe.html

Each :py:func:`~verticapy.vDataColumn` has its own catalog to save user modifications. In the previous example, we computed 
some aggregations for the ``is_booking`` column. Let's look at the catalog for that :py:func:`~verticapy.vDataColumn`:

.. ipython:: python

    expedia["is_booking"]._catalog

The catalog is updated whenever major changes are made to the data.

We can also view the vDataFrame's backend SQL code generation by setting the `sql_on` parameter to ``True`` with the :py:func:`~verticapy.set_option` function:

.. code-block:: python

    vp.set_option("sql_on", True)
    expedia["cnt"].describe()

.. code-block:: sql

    -- Computing the different aggregations
    SELECT
        /*+LABEL('vDataframe.aggregate')*/ 
        APPROXIMATE_COUNT_DISTINCT("cnt")
    FROM (
        SELECT
            "site_name",
            "posa_continent",
            "user_location_country",
            "user_location_region",
            "user_location_city",
            "orig_destination_distance",
            "user_id",
            "is_mobile",
            "is_package",
            "channel",
            "srch_adults_cnt",
            "srch_children_cnt",
            "srch_rm_cnt",
            "srch_destination_id",
            "srch_destination_type_id",
            "is_booking",
            "cnt",
            "hotel_continent",
            "hotel_country",
            "hotel_market",
            "hotel_cluster"
        FROM "public"."expedia"
    ) VERTICAPY_SUBTABLE
    LIMIT 1;

    -- Computing the descriptive statistics of all numerical columns using SUMMARIZE_NUMCOL
    SELECT
        /*+LABEL('vDataframe.describe')*/ 
        SUMMARIZE_NUMCOL("cnt") OVER ()
    FROM (
        SELECT
            "site_name",
            "posa_continent",
            "user_location_country",
            "user_location_region",
            "user_location_city",
            "orig_destination_distance",
            "user_id",
            "is_mobile",
            "is_package",
            "channel",
            "srch_adults_cnt",
            "srch_children_cnt",
            "srch_rm_cnt",
            "srch_destination_id",
            "srch_destination_type_id",
            "is_booking",
            "cnt",
            "hotel_continent",
            "hotel_country",
            "hotel_market",
            "hotel_cluster"
        FROM "public"."expedia"
    ) VERTICAPY_SUBTABLE;

.. ipython:: python
    :suppress:

    res = expedia["cnt"].describe()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_intro_vdf_describe_cnt.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_intro_vdf_describe_cnt.html

To control whether each query outputs its elasped time, use the ``time_on`` parameter of the :py:func:`~verticapy.set_option` function:

.. ipython:: python

    vp.set_option("sql_on", False)
    expedia = vp.vDataFrame("public.expedia") # creating a new vDataFrame to delete the catalog 
    vp.set_option("time_on", True)
    expedia.corr()

The aggregation's for each vDataColumn are saved to its catalog. If we again call the :py:func:`~verticapy.vDataFrame.corr` method, it'll complete in a couple seconds—the time needed to draw the graphic—because the aggregations have already been computed and saved during the last call:

.. ipython:: python

    start_time = time.time()
    expedia.corr();
    print("elapsed time = {}".format(time.time() - start_time))

To turn off the elapsed time and the SQL code generation options:

.. ipython:: python

    vp.set_option("sql_on", False)
    vp.set_option("time_on", False)

You can obtain the current :py:func:`~verticapy.vDataFrame` relation with the :py:func:`~verticapy.vDataFrame.current_relation` method:

.. ipython:: python

    print(expedia.current_relation())

The generated SQL for the relation changes according to the user's modifications. For example, if we impute the missing values of the ``orig_destination_distance`` vDataColumn by its average and then drop the ``is_package`` vDataColumn, these changes are reflected in the relation:

.. ipython:: python

    expedia["orig_destination_distance"].fillna(method = "avg");
    expedia["is_package"].drop();
    print(expedia.current_relation())

Notice that the ``is_package`` column has been removed from the ``SELECT`` statement and the ``orig_destination_distance`` is now using a ``COALESCE SQL`` function.

vDataFrame attributes and management
-------------------------------------

The :py:func:`~verticapy.vDataFrame` has many attributes and methods, some of which were demonstrated in the above examples. :py:func:`~verticapy.vDataFrame`s have two types of attributes:

- Virtual Columns (vDataColumn)
- Main attributes (columns, main_relation ...)

The vDataFrame's main attributes are stored in the ``_vars`` dictionary:

.. note:: You should never change these attributes manually.

.. ipython:: python

    expedia._vars

Data types
-----------

:py:func:`~verticapy.vDataFrame`s use the data types of its :py:func:`~verticapy.vDataColumn`s. The behavior of some :py:func:`~verticapy.vDataFrame` methods depend on the data type of the columns.

For example, computing a histogram for a numerical data type is not the same as computing a histogram for a categorical data type. 

The :py:func:`~verticapy.vDataFrame` identifies four main data types:

- ``int``: integers are treated like categorical data types 
    when their cardinality is low; otherwise, they are considered numeric
- ``float``: numeric data types
- ``date``: date-like data types (including timestamp)
- ``text``: categorical data types
 
Data types not included in the above list are automatically 
treated as categorical. You can examine the data types of 
the vDataColumns in a :py:func:`~verticapy.vDataFrame` using the 
:py:func:`~verticapy.vDataFrame.dtypes` method:

.. code-block:: python

    expedia.dtypes()

.. ipython:: python
    :suppress:

    res = expedia.dtypes()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_intro_vdf_expedia_dtypes.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_intro_vdf_expedia_dtypes.html

To convert the data type of a vDataColumn, use the :py:func:`~verticapy.vDataColumn.astype` method:

.. ipython:: python

    expedia["hotel_market"].astype("varchar");
    expedia["hotel_market"].ctype()

To view the category of a specific :py:func:`~verticapy.vDataColumn`, specify the :py:func:`~verticapy.vDataColumn` and use the :py:func:`~verticapy.vDataColumn.category` method:

.. ipython:: python

    expedia["hotel_market"].category()

Exporting, saving, and loading 
-------------------------------

The :py:func:`~verticapy.vDataFrame.save` and :py:func:`~verticapy.vDataFrame.load` functions allow you to save and load vDataFrames:

.. code-block:: python

    expedia.save()
    expedia.filter("is_booking = 1")

.. ipython:: python
    :suppress:

    expedia.save()
    res = expedia.filter("is_booking = 1")
    html_file = open("/project/data/VerticaPy/docs/figures/ug_intro_vdf_expedia_filter.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_intro_vdf_expedia_filter.html

To return a :py:func:`~verticapy.vDataFrame` to a previously saved structure, use the :py:func:`~verticapy.vDataFrame.load` function:

.. ipython:: python

    expedia = expedia.load();
    print(expedia.shape())

Because :py:func:`~verticapy.vDataFrame`s are views of data stored in the connected Vertica database, any modifications made to the :py:func:`~verticapy.vDataFrame` are not reflected in the underlying data in the database. To save a :py:func:`~verticapy.vDataFrame`'s relation to the database, use the :py:func:`~verticapy.vDataFrame.to_db` method.

It's good practice to examine the expected disk usage of the :py:func:`~verticapy.vDataFrame` before exporting it to the database:

.. code-block:: python

    expedia.expected_store_usage(unit = "Gb")

.. ipython:: python
    :suppress:

    res = expedia.expected_store_usage(unit = "Gb")
    html_file = open("/project/data/VerticaPy/docs/figures/ug_intro_vdf_expedia_storage_gb.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_intro_vdf_expedia_storage_gb.html

If you decide that there is sufficient space to store the :py:func:`~verticapy.vDataFrame` in the database, run the :py:func:`~verticapy.vDataFrame.to_db`  method:

.. code-block:: python
    
    expedia.to_db(
        "public.expedia_clean",
        relation_type = "table",
    )