.. _user_defined_funcs:

======================
User Defined Functions
======================


User-Defined Functions (UDFs) in VerticaPy offer 
a powerful mechanism for extending the 
functionality of the Vertica analytical database 
by allowing users to define custom functions 
tailored to their specific needs. These UDFs come 
in various types, each serving distinct purposes 
within the database ecosystem. 

- User-Defined Scalar Functions (UDSFs)
- User-Defined Analytic Functions (UDAF)
- User-Defined Aggregate Functions (UDAnF)
- User-Defined Transform Functions (UDTF)
- User-Defined Load Functions (UDL)
- User-Defined SQL Functions (UDF)

The table below summarizes the availability of
different types of UDx with their language support.

.. image:: /_static/img_udf_8.png

.. tab:: UDSF

    .. image:: /_static/img_udf_1.png

    User-Defined Scalar Functions are custom functions 
    that operate on individual values and return a 
    single value. These functions can be used in 
    SELECT statements, WHERE clauses, and various other 
    SQL expressions.

.. tab:: UDAF

    .. image:: /_static/img_udf_2.png

    User-Defined Analytic Functions perform calculations 
    across a set of rows related to the current row. 
    They are commonly used in analytical and windowed 
    queries, providing flexibility in data analysis.

.. tab:: UDAnF

    .. image:: /_static/img_udf_3.png

    User-Defined Aggregate Functions allow you to define 
    custom aggregate operations on groups of rows. 
    These functions are particularly useful in 
    summarizing and aggregating data.

.. tab:: UDTFs

    .. image:: /_static/img_udf_4.png

    User-Defined Transform Functions are generally used 
    during the loading process to transform or preprocess 
    data before it is loaded into the Vertica database. 
    They are essential for data cleansing and 
    transformation tasks.

.. tab:: UDL

    .. image:: /_static/img_udf_5.png

    User-Defined Load Functions are used to define how 
    Vertica should load data into a table. They allow 
    customization of the loading process, enabling 
    users to handle specific data formats or perform 
    additional processing during data loading.

.. tab:: UDF

    .. image:: /_static/img_udf_6.png

    User-Defined SQL Functions enable the creation of 
    custom SQL functions with specific logic. These 
    functions can be invoked in SQL queries, providing 
    a way to encapsulate complex logic in a modular and 
    reusable manner.

----

Unfenced Mode
^^^^^^^^^^^^^

Vertica also has an **Unfenced Mode** for advanced users.

.. image:: /_static/img_udf_7.png

In Fenced Mode, stability is emphasized as it employs a 
separate zygote process. This separation ensures that 
User-Defined Extension (UDx) crashes do not adversely 
impact the core Vertica process.

| ✓ **Pro:** :bdg-success:`Protected`
| ✗ **Con:** :bdg-danger:`Slower`


Conversely, in Unfenced Mode, the safety of zygote 
processes is not guaranteed. However, it excels in 
performance as data does not need to move back and forth.


| ✓ **Pro:** :bdg-success:`Faster`
| ✗ **Con:** :bdg-danger:`Unprotected`

Let's have a look at some use-cases to see the difference
in speed for both the modes:

.. tab:: UDx XGBoost Scoring

    **Size:** 

    .. list-table:: 
        :header-rows: 1

        * - Rows
          - Columns
        * - 31.29M
          - 2559

    **Query:**

    .. code-block:: SQL

        SELECT xgb_score(...) 
        FROM s_table;

    .. ipython:: python
        :suppress:

        import plotly.graph_objects as go
        labels = ['Unfenced', 'Fenced']
        heights = [4*60+23, 14*60+39]
        colors = ['red', 'green']
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict({"color": "#888888"}),
        )
        for label, height, color in zip(labels, heights, colors):
            fig.add_trace(go.Bar(
                x=[label],
                y=[height],
                marker_color=color,
                text=[height],
                textposition='outside',
                name=label,
            ))
        fig.update_layout(
        title='31.29 M rows & 2559 columns',
        yaxis=dict(title='Execution Time (secs)'),
        bargap=0.2,
        width = 400,
        height = 500
        )
        fig.write_html("SPHINX_DIRECTORY/figures/user_defined_functions_udx_xgboost_scoring.html")

    .. raw:: html
      :file: SPHINX_DIRECTORY/figures/user_defined_functions_udx_xgboost_scoring.html

.. tab:: Top K per partition

    **Size:** 

    .. list-table:: 
        :header-rows: 1

        * - Rows
          - Columns
        * - 100M
          - 2


    **Query:**

    .. code-block:: SQL

        SELECT COUNT(*) 
        FROM 
            (
                SELECT polyk(10, x, y) 
                OVER (PARTITION BY x) FROM foo
            ) AS P;


    .. ipython:: python
        :suppress:

        import plotly.graph_objects as go
        labels = ['Unfenced', 'Fenced']
        heights = [1076, 1187]
        colors = ['red', 'green']
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict({"color": "#888888"}),
        )
        for label, height, color in zip(labels, heights, colors):
            fig.add_trace(go.Bar(
                x=[label],
                y=[height],
                marker_color=color,
                text=[height],
                textposition='outside',
                name=label,
            ))
        fig.update_layout(
        title='100M rows & 2 columns',
        yaxis=dict(title='Execution Time (ms)'),
        bargap=0.2,
        width = 400,
        height = 500
        )
        fig.write_html("SPHINX_DIRECTORY/figures/user_defined_functions_udx_top_k.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/user_defined_functions_udx_top_k.html

.. tab:: Shorter String

    
    **Size:** 

    .. list-table:: 
        :header-rows: 1

        * - Rows
          - Columns
        * - 100M
          - 2

    **Query:**

    .. code-block:: SQL

        SELECT COUNT(*) 
        FROM foo 
        WHERE shorterString(a,b) = a;


    .. ipython:: python
        :suppress:

        import plotly.graph_objects as go
        labels = ['Unfenced', 'Fenced']
        heights = [76868, 82638]
        colors = ['red', 'green']
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict({"color": "#888888"}),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict({"color": "#888888"}),
        )
        for label, height, color in zip(labels, heights, colors):
            fig.add_trace(go.Bar(
                x=[label],
                y=[height],
                marker_color=color,
                text=[height],
                textposition='outside',
                name=label,
            ))
        fig.update_layout(
        title='100M rows & 2 columns',
        yaxis=dict(title='Execution Time (ms)'),
        bargap=0.2,
        width = 400,
        height = 500
        )
        fig.write_html("SPHINX_DIRECTORY/figures/user_defined_functions_udx_shorter_string.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/user_defined_functions_udx_shorter_string.html

----

Example - XGB Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^


Let us look at an exmaple of a UDx 
performing XGB Prediction. The elements
of code are explained in the image;


.. image:: /_static/img_udf_9.png

The above example was using ``python``.
Let us look at the same function using
``SQL``:

.. image:: /_static/img_udf_10.png


.. note::

    For more information, please refer to the
    `Python SDK doc <https://docs.vertica.com/23.4.x/en/extending/developing-udxs/developing-with-sdk/python-sdk/>`_

----

Auto-Lambda Generator Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Understanding how VerticaPy User-Defined Extensions 
(UDx) work involves exploring the seamless process 
of creating and deploying custom functions within 
Vertica. As an illustrative example, consider the 
auto-generation of a lambda function.

.. image:: /_static/img_udf_11.png

Once a user creates a custom UDx using a Python 
function, the workflow becomes remarkably 
straightforward. The user interacts solely with the 
Python function, and VerticaPy takes care of the 
rest. VerticaPy intelligently parses the provided 
Python function, creates the necessary User-Defined 
Functions (UDFs), and efficiently installs them 
across different nodes in the Vertica cluster. 
This installation process ensures that the custom 
functionality is distributed across multiple nodes, 
facilitating load distribution and enhancing 
scalability. By automating the parsing, creation, 
and distribution of UDx, VerticaPy streamlines 
the development and deployment of custom functions, 
allowing users to focus on the logic and 
functionality of their Python code without worrying 
about the intricacies of distributed computing 
within the Vertica environment

Example
--------


Let's delve into an example of a User-Defined 
Extension (UDx) function, specifically focusing 
on the ``math.isClose()`` function. When a user 
defines this new function using the 
``create_lib_udf`` function in VerticaPy, the 
process becomes exceptionally streamlined. 
VerticaPy, behind the scenes, automatically 
generates the corresponding lambda function in 
Python. Furthermore, it seamlessly installs this 
function across the different nodes in the 
Vertica cluster using SQL commands.

.. image:: /_static/img_udf_12.png

The beauty of this approach shines when the 
``math.isClose()`` function is called. As part 
of the Vertica cluster, each node independently 
implements the function. Leveraging the power of 
distributed computing, the aggregate result is 
then efficiently calculated across all nodes. 
This ensures that the ``math.isClose()`` function, 
once defined and installed, seamlessly integrates 
into the Vertica environment, providing users with 
a scalable and distributed solution for their 
mathematical closeness calculations.

For more in-depth detail on the uses of User-Defined
Functions please refer to 
:ref:`user_guide.full_stack`.

.. seealso::

    | SDK Documentation: `Vertica Python SDK <https://www.vertica.com/docs/10.1.x/HTML/PythonSDK/sdk_documentation.html>`_
    | Notebooks on UDFx: :ref:`user_guide.full_stack`
    | Vertica Documentation on UDx: `Python SDK doc <https://docs.vertica.com/23.4.x/en/extending/developing-udxs/developing-with-sdk/python-sdk/>`_