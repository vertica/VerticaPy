.. _user_guide.data_ingestion:

===============
Data Ingestion
===============

VerticaPy supports the ingestion of data in the following formats into Vertica:

- CSV
- JSON
- Parquet
- SHP
- ORC
- Avro
- pandas DataFrame

You can use the :py:func:`~verticapy.read_file` function to ingest all the above file types except pandas DataFrames and SHP, which instead use file-specific ingestion functions - :py:func:`~verticapy.read_pandas` and :py:func:`~verticapy.read_shp`. There are also file-specifc ingestion functions for JSON, Avro, and CSV files that use flex tables to ingest the data.

Unless you specify the columns' data types with the dtype parameter, the ingestion functions automatically predict the data type of each column. If you provide the column data types, the function does not need to parse the file and predict data types, likely increasing ingestion speed and precision.

For the :py:func:`~verticapy.read_file` function, if the file to ingest is located in the Vertica database, you must provide the column data types with the dtype parameter.

.. note:: 
    
    As performance optimizations made in the Vertica database are carried over to VerticaPy, try to optimize the structure of your projections for new tables in Vertica.

In the following sections, we will explore a few of the ingestion functions and some of the options they support.

Ingest files with :py:func:`~verticapy.read_file`
-------------------------------------------------

The :py:func:`~verticapy.read_file` function inspects and ingests files in any of the following formats:

- CSV
- Parquet
- ORC
- JSON
- Avro

Some of the supported function options include:

- ``dtype``: provide a dictionary of data types for the columns, where the keys are the column names and the values are the column data types. The data types in the dictionary replace the automatically predicted data types.
- ``insert``: if set to ``True``, the data is ingested into the relation specified by the ``table_name`` and, optionally, the ``schema`` parameters.
- ``table_name``: specifies the name of the table to create in the Vertica database, or the name of the table to which the data is inserted.
- ``temporary_table``: if set to ``True``, creates a temporary table.
- ``ingest_local``: if set to ``True``, the file is ingested from the local machine. In this case, the ``dtypes`` parameter is optional; if no value is provided, the function predicts the data type of each column.
- ``genSQL``: if set to ``True``, the function generates the SQL code the creates the table but does not execute it. This is a convenient way to check the final relation types.

For a full list of supported options, see the documentation or use the :py:func:`~verticapy.help` function.

.. note::

    All data files used in this tutorial are availble in the VerticaPy datasets directory. For demo purposes, the following examples ingest the data files using :py:func:`~verticapy.read_file` and other file-specific read functions. However, VerticaPy includes a set of dataset loading functions that allow you to easily ingest the data files in the datasets directory.

In the following examples, we will demonstrate how to use the :py:func:`~verticapy.read_file` function to ingest data into Vertica. Both file location options, in-database and local, will be explored.

Let's begin with the case where the file is located in the database. We'll ingest the iris.csv file, a popular classification dataset. First, before we ingest the file, run the function with the ``genSQL`` parameter set to ``True`` to view the SQL that will be used to create the table. Because the file is located in the database, we must specify the data types for each column with the ``dtypes`` parameter:

.. note:: 

    For the examples in this tutorial, replace ``path-to-file`` in the path parameter with the ``path`` to the file in your Vertica database or local machine.

.. code-block:: python

    import verticapy as vp

    vp.read_file(
        path = "path-to-file/iris.csv",
        dtype = {
            "Id": "Integer",
            "SepalLengthCm": "Numeric",
            "SepalWidthCm": "Numeric",
            "PetalLengthCm": "Numeric",
            "PetalWidthCm": "Numeric",
            "Species": "Varchar(20)",
        },
        schema = "public",
        genSQL = True,
    )

To ingest the file into Vertica, remove the ``genSQL`` parameter from the above command and rerun the function:

.. note:: If no table name is specified by ``table_name`` parameter, the name of the file is used for the table name.

.. code-block:: python

    vp.read_file(
        path = "path-to-file/iris.csv",
        dtype = {
            "Id": "Integer",
            "SepalLengthCm": "Numeric",
            "SepalWidthCm": "Numeric",
            "PetalLengthCm": "Numeric",
            "PetalWidthCm": "Numeric",
            "Species": "Varchar(20)",
        },
        schema = "public",
    )

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_iris
    iris = load_iris()
    res = iris
    html_file = open("SPHINX_DIRECTORY/figures/user_guide_data_ingestion_iris.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/user_guide_data_ingestion_iris.html

When the file to ingest is not located on your local machine, and is on the server instead, then you must set the ``ingest_local`` parameter to ``False``. 

``ingest_local`` is ``True`` by default.

.. note:: In some cases where the CSV file has a very complex structure, local ingestion might fail. If this occurs, you will have to move the file into the database and then ingest the file from that location.

.. code-block::

    vp.read_file(
        path = "path-to-file/iris.csv",
        schema = "public",
        table_name = "iris_local",
        ingest_local = False,
    )

To ingest multiple files of the same type, use the following 
syntax in the path parameter (in this case for multiple CSV files): ``path = "path-to-files/*.csv"``

Ingest CSV files
----------------

In addition to :py:func:`~verticapy.read_file`, you can also ingest CSV files with the :py:func:`~verticapy.read_csv` function, which ingests the file using flex tables. This function provides options not available in :py:func:`~verticapy.read_file`, such as:

- ``sep``: specify the column separator.
- ``parse_nrows``: the function creates a file of nrows from the data file to identify 
the data types. This file is then dropped and the entire data file is ingested. If your data file is large, this data type inference process should speed up the file ingestion speed.
- ``materialize``: if set to ``True``, the flex table used to ingest the data file is materialized into a table; otherwise, the data remains in a flex table.

For a full list of supported options, see :py:func:`~verticapy.read_csv` or use the :py:func:`~verticapy.help` function.

In the following example, we will use :py:func:`~verticapy.read_csv` to ingest a subset of the Titanic dataset. To begin, load the entire Titanic dataset using the :py:func:`~verticapy.datasets.load_titanic` function:

.. ipython:: python

    from verticapy.datasets import load_titanic

    titanic = load_titanic()

To convert a subset of the dataset to a CSV file, select the desired rows in the dataset and use the :py:func:`~verticapy.vDataFrame.to_csv` :py:mod:`~verticapy.vDataFrame` method:

.. ipython:: python

    titanic[0:50].to_csv(
        path = "titanic_subset.csv",
    )

Before ingesting the above CSV file, we can check its columns and their data types with the :py:func:`~verticapy.pcsv` function:

.. ipython:: python

    vp.pcsv(
        path = "titanic_subset.csv",
        sep = ",",
        na_rep = "",
    )

Now, setting the ``ingest_local`` parameter to ``True``, ingest the CSV file into the Vertica database:

.. code-block:: python

    vp.read_csv(
        "titanic_subset.csv",
        schema = "public",
        table_name = "titanic_subset",
        sep = ",",
        ingest_local = True,
    )
   
If we want to insert additional data from the original Titanic dataset into the ``public.titanic_subset`` table, we can do so by setting the ``insert`` parameter of the :py:func:`~verticapy.read_csv` function to ``True``:

.. hint:: You can also insert data into an existing Vertica table with the :py:func:`~verticapy.insert_into` function.

.. code-block:: python

    titanic[50:100].to_csv(
        path = "titanic_more_data.csv",
    )

    vp.read_csv(
        "titanic_more_data.csv",
        schema = "public",
        table_name = "titanic_subset",
        sep = ",",
        insert = True,
    )

Ingest JSON files
------------------

As with CSV files, VerticaPy provides a file-specific ingestion function for JSON files, :py:func:`~verticapy.read_json`, which supports additional options, including:

- ``usecols``: provide a list of JSON parameters to ingest. Other JSON parameters are ignored.
- ``start_point``: name the key in the JSON load data at which to begin parsing
- ``flatten_maps``: set whether sub-maps within the JSON data are flattened.
- ``materialize``: if set to ``True``, the flex table used to ingest the data is materialized into a table.

For a full list of supported options, see the :py:func:`~verticapy.read_json` or use the :py:func:`~verticapy.help` function.

VerticaPy also provides a :py:func:`~verticapy.pjson` function to parse JSON files to identify columns and their respective data types.

In the following example, we load the iris dataset using the :py:func:`~verticapy.datasets.load_iris` dataset, convert the vDataFrame to JSON format with the :py:func:`~verticapy.to_json` method, then ingest the JSON file into Vetica:

.. code-block:: python

    from verticapy.datasets import load_iris

    iris = load_iris()
    iris.to_json(
        path = "iris_local.json",
    )
    vp.read_json(
        path = "iris_local.json",
        table_name = "iris_ingest",
        schema = "public",
    )

Other file types
-----------------

For more information about other file-specific ingestion functions, see the following reference pages, which include examples:

- pandas DataFrames: :py:func:`~verticapy.read_pandas`
- Avro: :py:func:`~verticapy.read_avro`
- SHP: :py:func:`~verticapy.read_shp`