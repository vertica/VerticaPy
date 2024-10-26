.. _user_guide.introduction.best_practices:

Best practices
===============

In this tutorial, we will explore some best practices and optimizations to help you get the most out of Vertica and VerticaPy.

Restrict objects and operations to essential columns 
-------------------------------------------------------

As VerticaPy is effectively an abstraction of SQL, any database-level optimizations you make in your Vertica database carry over to VerticaPy. In Vertica, optimization is centered on projections, which are collections of table columns—from one or more tables—stored on disk in a format that optimizes query execution. When you write queries in terms of the original tables, the query uses the projections to return query results. For details about creating and designing projections, see the Projections section in the Vertica documentation.

Projections are created and managed in the Vertica database, but you can leverage the power of projections in VerticaPy with features such as the :py:mod:`~verticapy.vDataFrame` usecols parameter, which specifies the columns from the input relation to include in the :py:mod:`~verticapy.vDataFrame`. As columnar databases perform better when there are fewer columns in the query, especially when you are working with large datasets, limiting :py:mod:`~verticapy.vDataFrame` and operations to essential columns can lead to a significant performance improvement. By default, most :py:mod:`~verticapy.vDataFrame` methods use all numerical columns in the :py:mod:`~verticapy.vDataFrame`, but you can restrict the operation to specific columns.

In the following examples, we'll demonstrate how to create a :py:mod:`~verticapy.vDataFrame` from specific columns in the input relation, and then run methods on that :py:mod:`~verticapy.vDataFrame`. First, load the titanic dataset into Vertica using the :py:func:`~verticapy.datasets.load_titanic` function:

.. code-block:: python

    from verticapy.datasets import load_titanic

    load_titanic()

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_titanic
    res = load_titanic()
    html_file = open("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_laod_titanic.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_laod_titanic.html

Supposing we are only interested in the ``survived``, ``pclass``, ``age``, ``parch``, and ``sibsp`` columns, we can create a vDataFrame with just those columns by specifying them in the usecols parameter:

.. code-block:: python
    
    import verticapy as vp

    titanic = vp.vDataFrame(
        "public.titanic",
        usecols = ["survived", "pclass", "age", "parch", "sibsp"],
    )
    titanic.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    titanic = vp.vDataFrame(
        "public.titanic",
        usecols = ["survived", "pclass", "age", "parch", "sibsp"],
    )
    res = titanic.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_laod_titanic_selective.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_laod_titanic_selective.html

If we run the :py:func:`~verticapy.vDataFrame.avg` method without specifying columns, all numerical vDataFrame columns are included in the operation:

.. note:: To examine the generated SQL for each command, turn on the "sql_on" option using :py:func:`~verticapy.set_option`.

.. ipython:: python
    
    # Turning on SQL.
    vp.set_option("sql_on", True)

    titanic.avg()
    
To turn off the SQL code generation option:

.. ipython:: python
    
    # Turning off SQL.
    vp.set_option("sql_on", False)

To restrict the operation to specific columns in the :py:mod:`~verticapy.vDataFrame`, provide the column names in the ``columns`` parameter:

.. code-block:: python

    titanic.avg(columns = ["age", "survived"])

.. ipython:: python
    :suppress:

    res = titanic.avg(columns = ["age", "survived"])
    html_file = open("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_titanic_avg.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_titanic_avg.html

As we are working with a small dataset, the perfomance impact of excluding unncessary columns is not very significant. However, with large datasets (e.g. greater than a TB), the impact is much greater, and choosing essential columns becomes a key step in improving performance.

Instead of specifying essential columns to include, some methods allow you to list the columns to exclude with the ``exclude_columns`` parameter:

.. ipython:: python

    titanic.numcol(exclude_columns = ["parch", "sibsp"])

.. note:: 

    To list all columns in a :py:mod:`~verticapy.vDataFrame`, including non-numerical columns, use the :py:func:`~verticapy.vDataFrame.get_columns` method.

You can then use this truncated list of columns in another method call; for instance, to compute a correlation matrix:

.. code-block:: python

    titanic.corr(columns = titanic.numcol(exclude_columns = ["parch", "sibsp"]))

.. ipython:: python
    :suppress:

    vp.set_option("plotting_lib", "plotly")
    fig = titanic.corr(columns = titanic.numcol(exclude_columns = ["parch", "sibsp"]))
    fig.write_html("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_titanic_corr.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_titanic_corr.html

Save the current relation
--------------------------

The :py:mod:`~verticapy.vDataFrame` works like a ``view``, a stored query that encapsulates one or more SELECT statements. 
If the generated relation uses many different functions, the computation time for each method call is greatly increased.

Small transformations don't drastically slow down computation, but heavy transformations (multiple joins, frequent use of advanced analytical funcions, moving windows, etc.) can result in noticeable slowdown. When performing computationally expensive operations, you can aid performance by saving the vDataFrame structure as a table in the Vertica database. We will demonstrate this process in the following example.

First, create a :py:mod:`~verticapy.vDataFrame`, then perform some operations on that :py:mod:`~verticapy.vDataFrame`:

.. code-block:: python

    titanic = vp.vDataFrame("public.titanic")
    titanic["sex"].label_encode()["boat"].fillna(method = "0ifnull")["name"].str_extract(
        ' ([A-Za-z]+)\.').eval("family_size", expr = "parch + sibsp + 1").drop(
        columns = ["cabin", "body", "ticket", "home.dest"])["fare"].fill_outliers().fillna()

.. ipython:: python
    :suppress:

    titanic = vp.vDataFrame("public.titanic")
    titanic["sex"].label_encode()["boat"].fillna(method = "0ifnull")["name"].str_extract(' ([A-Za-z]+)\.').eval("family_size", expr = "parch + sibsp + 1").drop(columns = ["cabin", "body", "ticket", "home.dest"])["fare"].fill_outliers().fillna()

.. ipython:: python

    print(titanic.current_relation())

To understand how Vertica executes the different aggregations in the above relation, let's take a look at the query plan:

.. note:: python

    Query plans can be hard to interpret if you don't know how to parse them. For more information, see `query plan information and structure <https://docs.vertica.com/24.1.x/en/admin/managing-queries/query-plans/query-plan-information-and-structure/>`_.

.. ipython:: python

    print(titanic.explain())

Looking at the plan and its associated relation, it's clear that the transformations we applied to the vDataFrame result in a complicated relation. 

Each method call to the :py:mod:`~verticapy.vDataFrame` must use this relation for computation. 

.. note:: 

    To better understand your queries, check out the :ref:`~verticapy.performance.vertica.qprof.QueryProfiler` function.

To save the relation as a table in the Vertica and replace the current relation in VerticaPy with the new table relation, use the :py:func:`~verticapy.vDataFrame.to_db` method with the ``inplace`` parameter set to True:

.. code-block:: python

    vp.drop(
        "public.titanic_clean",
        method = "table",
    ) # drops any existing table with the same schema and name
    titanic.to_db("public.titanic_clean",
        relation_type = "table",
        inplace = True,
    )

.. ipython:: python
    :suppress:

    vp.drop(
        "public.titanic_clean",
        method = "table",
    ) # drops any existing table with the same schema and name
    titanic.to_db(
        "public.titanic_clean",
        relation_type = "table",
        inplace = True,
    )

.. ipython:: python
    
    print(titanic.current_relation())

When dealing with very large datasets, it's best to take caution before saving relations with complicated transformations. Ideally, you will perform a thorough data exploration, and only execute heavy transformations when essential.

Use the help function
----------------------

For a quick and convenient way to view information about an object or function, use the :py:func:`help` function:

.. ipython:: python

    help(vp.connect)

Close your connections
-----------------------

Each connection to the database increases the concurrency on the system, so try to close connections when you're done with them. VerticaPy simplifies the connection process by allowing the user to create an auto-connection, but the closing of connections must be done manually with the :func:`~verticapy.close_connection` function.

To demonstrate, create a database connection:

.. code-block:: python

    vp.connect("VerticaDSN")

When you are done making changes, close the connection with the :func:`~verticapy.close_connection` function:

.. code-block:: python

    vp.close_connection()

It is especially important to close connections when you are working in an environment with mutliple users.

Consider a method's time complexity
--------------------------------------

Some techniques are significantly more computationally expensive than others. For example, a Kendall correlation is very expensive compared to a Pearson correlation because, unlike Pearson, Kendall correlations use a cross join, resulting in a time complexity of O(n*n) (where n is the number of rows). 

Let's compare the time needed to compute these two correlations on the ``titanic`` dataset:

.. ipython:: python

    import time

    titanic = vp.vDataFrame("public.titanic")
    start_time = time.time()
    x = titanic.corr(method = "pearson", show = False)
    print("Pearson, time: {0}".format(time.time() - start_time))
    start_time = time.time()
    x = titanic.corr(method = "kendall", show = False)
    print("Kendall, time: {0}".format(time.time() - start_time))

Limit plot elements
--------------------

Graphics are an essential tool to understand your data, but they can become difficult to parse if they contain 
too many elements. VerticaPy provides options that restrict plots to specified elements. To demonstrate, let's first draw a multi-histogram with a categorical column with thousands of categories:

.. code-block:: python

    titanic.bar(["name", "survived"])

.. ipython:: python
    :suppress:

    fig = titanic.bar(["name", "survived"], width = 900)
    fig.write_html("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_titanic_bar_plot.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_titanic_bar_plot.html

VerticaPy outputs the bar chart, but the number of categories makes the graph basically incomprehensible. Instead, whenever possible, try to create graphics with as few categories as possible for your use case:

.. code-block:: python

    titanic.hist(["pclass", "survived"])

.. ipython:: python
    :suppress:

    fig = titanic.hist(["pclass", "survived"])
    fig.write_html("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_titanic_hist_plot.html")

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_titanic_hist_plot.html

To view the cardinality of your variables, use the :ref:`~verticapy.vDataFrame.nunique` method:

.. code-block:: python

    titanic.nunique()

.. ipython:: python
    :suppress:

    res = titanic.nunique()
    html_file = open("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_nunqiue.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_nunqiue.html

Filter unnecessary data
------------------------

Filtering your data is a crucial step in data preparation. Proper filtering avoids unnecessary computations and greatly 
improves the performance of each method call. While the performance impact can be minimal for small datasets, filtering large datasets is key to improving performance.

For example, if we are only interested in analyzing Titanic passengers who didn't have a lifeboat, we can filter on this requirement using the :ref:`~verticapy.vDataFrame.filter` method: 

.. code-block:: python

    titanic.filter("boat IS NOT NULL")

.. ipython:: python
    :suppress:

    res = titanic.filter("boat IS NOT NULL")
    html_file = open("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_filter.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_filter.html

To drop unnecessary columns from your vDataFrame, use the :func:`~verticapy.vDataFrame.drop` method:

.. code-block:: python

    titanic.drop(["name", "body"])

.. ipython:: python
    :suppress:

    res = titanic.drop(["name", "body"])
    html_file = open("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_drop_name_body.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_drop_name_body.html

The dropped columns are excluded from the relation's generated ``SELECT`` query:

.. ipython:: python

    print(titanic.current_relation())

Maximize your resources
------------------------

Large datasets often contain hundreds of columns. These datasets require VerticaPy to compute many 
concurrent, resource-intensive aggregations. To limit the impact of these aggregations, you can control the number of queries that VerticaPy sends to the system, which allows for some useful optimizations.

In the following example, we'll explore a couple of these optimizations. First, generate a dataset:

.. code-block:: python

    from verticapy.datasets import gen_dataset

    vp.drop("public.test_dataset", method = "table") # drop an existing table with the same schema and name
    features_ranges = {}
    for i in range(20):
        features_ranges[f"x{i}"] = {"type": float, "range": [0, 1]}
    vp.drop("test_dataset", method = "table")
    vdf = gen_dataset(
        features_ranges,
        nrows = 100000,
    ).to_db(
        "test_dataset", 
        relation_type = "table", 
        inplace = True,
    )
    vdf.head(100)

.. ipython:: python
    :suppress:

    from verticapy.datasets import gen_dataset

    vp.drop("public.test_dataset", method = "table") # drop an existing table with the same schema and name
    features_ranges = {}
    for i in range(20):
        features_ranges[f"x{i}"] = {"type": float, "range": [0, 1]}
    vp.drop("test_dataset", method = "table")
    vdf = gen_dataset(
        features_ranges,
        nrows = 100000,
    ).to_db(
        "test_dataset", 
        relation_type = "table", 
        inplace = True,
    )
    res = vdf.head(100)
    html_file = open("SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_gen_dataset.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_introduction_best_practices_gen_dataset.html

To monitor how VerticaPy is computing the aggregations, use the :py:func:`~verticapy.set_option` function to turn on SQL code generation and turn off cache:

.. ipython:: python

    vp.set_option("sql_on", True)
    vp.set_option("cache", False)

VerticaPy allows you to send multiple queries, either iteratively or concurrently, to the database when computing aggregations.

First, let's send a single query to compute the average for all columns in the :py:mod:`~verticapy.vDataFrame`:

.. ipython:: python

    display(vdf.avg(ncols_block = 20))

We see that there was one SELECT query for all columns in the :py:mod:`~verticapy.vDataFrame`. 
You can reduce the impact on the system by using the ``ncols_block`` parameter to split the computation into multiple iterative queries, where the value of the parameter is the number of columns included in each query.

For example, setting ``ncols_block`` to 5 will split the computation, which consists of 20 total columns, into 4 separate queries, each of which computes the average for 5 columns:

.. ipython:: python

    display(vdf.avg(ncols_block = 5))

In addition to spliting up the computation into separate queries, you can send multiple queries to the database concurrently. 
You specify the number of concurrent queries with the ``processes`` parameter, which defines the number of workers involved in the computation. Each child process creates a DB connection and then sends its query. In the following example, we use 4 ``processes``:

.. code-block:: python

    vdf.avg(ncols_block = 5, processes = 4)