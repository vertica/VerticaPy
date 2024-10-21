.. _user_guide.full_stack.udf:

=======================
User-Defined Functions
=======================

Vertica is an analytical database with many advanced functions. While Vertica obviously doesn't have every function, it comes pretty close by implementing a Python SDK that allows users to create and import their own "lambda" functions into Vertica.

While the SDK is somewhat difficult to use, VerticaPy makes it easy by generating UDFs from python code and importing them into Vertica for you.

To demonstrate this, let's look at the following example. We want to bring into Vertica two functions from the 'math' module. We also want to import our own custom 'normalize_titanic' function defined in the 'pmath.py' file.

.. ipython:: python

    import os
    import math

    import verticapy as vp
    from verticapy.sdk.vertica.udf import generate_lib_udf

    def normalize_titanic(age, fare):
        return (age - 30.15) / 14.44, (fare - 33.96) / 52.65

    # The generated files must be placed in the folder: /home/dbadmin/
    file_path = "/home/dbadmin/python_math_lib.py"
    pmath_path = os.path.dirname(vp.__file__) + "/tests/udf/pmath.py"
    udx_str, udx_sql = generate_lib_udf(
        [
            (
                math.exp,
                [float],
                float,
                {},
                "python_exp",
            ),
            (
                math.isclose,
                [float, float],
                bool,
                {"abs_tol": float},
                "python_isclose",
            ),
            (
                normalize_titanic,
                [float, float],
                {
                    "norm_age": float,
                    "norm_fare": float,
                },
                {},
                "python_norm_titanic",
            ),
        ],
        library_name = "python_math",
        include_dependencies = pmath_path,
        file_path = file_path,
        create_file = False,
    )

We simply need to provide some information about our functions (input/output types, parameters, library name, etc.) and VerticaPy will generate two files.

One for the UDx:

.. ipython:: python

    print(udx_str)

And one for the SQL statements.

.. ipython:: python

    print("\n".join(udx_sql))

We can then run these queries in our server.

Our functions are now available in Vertica.

.. code-block:: ipython

    %load_ext verticapy.sql
    %sql -c "SELECT python_exp(1);"

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    res = vp.vDataFrame("SELECT EXP(1) AS python_exp;")
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_udf_1.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_udf_1.html

.. code-block:: ipython

    %sql -c "SELECT python_isclose(2, 3 USING PARAMETERS abs_tol=0.01);"

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    res = vp.vDataFrame("SELECT False AS python_isclose;")
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_udf_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_udf_2.html

.. code-block:: ipython

    %sql -c "SELECT python_isclose(2, 3 USING PARAMETERS abs_tol=1);"

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    res = vp.vDataFrame("SELECT True AS python_isclose;")
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_udf_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_udf_3.html

.. code-block:: ipython

    %sql -c "SELECT python_norm_titanic(20.0, 30.0) OVER();"

.. ipython:: python
    :suppress:
    :okwarning:

    import verticapy as vp
    res = vp.vDataFrame("SELECT -0.702908587257618 AS norm_age, -0.0752136752136752 AS norm_fare;")
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_table_udf_4.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_table_udf_4.html

It is now easy to bring customized Python functions.