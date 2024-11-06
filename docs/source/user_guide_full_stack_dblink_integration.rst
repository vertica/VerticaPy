.. _user_guide.full_stack.db_link:

=====================
DBLINK in VerticaPy
=====================

Introduction
-------------

Starting with VerticaPy 0.12.0, you can work with other databases, such as ``PostgreSQL`` and ``mySQL``, using ``DBLINK`` functionality. ``DBLINK`` is a Vertica User Defined Transform Function coded in ``C++`` that runs ``SQL`` against other databases. To setup and learn more about DBLINK in Vertica, please view the 
`github repo <https://github.com/vertica/dblink>`_.

In order to use this new functionality, we first need to install the ``ODBC`` driver and manager, as well as configure ``DBLINK`` on all nodes of the cluster. Configuration entails three files:

- ``dblink.cids``
- ``odbc.ini``
- ``odbcinst.ini``

These files provide the host server address, username, and password, as well as the database name that we want to access. In future versions, we are planning to simplify this process and automate the creation of these files. 

In the next section, let's work through an example of a database in ``PostgreSQL``.

Connecting to an External Database
-----------------------------------

.. ipython:: python

    # Importing VerticaPy
    import verticapy as vp

We first need to provide the connection information that we have set up in the Connection Identifier Database file (``dblink.cids``). We can select a special character symbol to identify this connection.

Let's try to set up a connection with an external ``PostgreSQL`` database, which we name ``pgdb``. The connection details for ``pgdb``, including server name, user name etc., are in the configuration files mentioned in the introduction section.

.. code-block:: python

    # Setting up a connection with a database with the alias "pgdb"
    vp.set_external_connection(
        cid = "pgdb",
        rowset = 500,
        symbol = "&",
    )

Creating a :py:mod:`~verticapy.vDataFrame`
-------------------------------------------

We can create a :py:mod:`~verticapy.vDataFrame` from a table stored in an external 
database by setting the ``external`` parameter to ``True``. 

SQL can be used to fetch required data, and we can provide an identifying symbol that can be used for fetching perform queries with SQL.

.. code-block:: python

    # Creating a vDataFrame using an SQL query by setting external = True.
    tab_data = vp.vDataFrame(
        input_relation = "airports",
        external = True,
        symbol = "&",
    )
    tab_data.head(100)

.. ipython:: python
    :suppress:

    import verticapy as vp
    vp.drop("public.airports")
    tab_data = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/flights/airports.csv",
        schema = "public",
        table_name = "airports")
    res = tab_data
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_table.html

All :py:mod:`~verticapy.vDataFrame` methods are available for this imported table. 

For example, we can get all the column names:

.. ipython:: python

    # Get all columns of the dataset
    tab_data.get_columns()

Or the column data types:

.. ipython:: python

    # Get data types of all columns inside the dataset
    tab_data.dtypes()

Or the count of the datapoints:

.. code-block:: python

    # Counting all elements inside each column
    tab_data.count()

.. note::

    Every time we perform these calculations or call the :py:mod:`~verticapy.vDataFrame`, it runs the SQL query to fetch all the data from the external database. After retrieving the entire table, the operations are computed by Vertica. In order to push the queries to a remote database, we can use the option ``sql_push_ext``. When we create a :py:mod:`~verticapy.vDataFrame` with this option activated, all the aggregations are done on the external database using SQL.

.. code-block:: python

    # Creating a vDataFrame and setting sql_push_ext to True, which tries 
    # to push SQL queries to external database (where possible).
    Ext_Table = vp.vDataFrame(
        input_relation = "airports",
        external = True,
        symbol = "&",
        sql_push_ext = True,
    )
    Ext_Table.head(100)

If we look at the SQL generated in background, we can see that 
it pushes the aggregation query to the database.

.. code-block:: python

    # Turning on SQL output to view the queries
    vp.set_option("sql_on",True)

Let's look at the count query again, and see how VerticaPy is pushing it to the external database.

.. code-block:: python

    # Counting elements in each column
    Ext_Table.count()

.. code-block:: sql

    SELECT
    DBLINK(USING PARAMETERS cid='pgdb', query='
        SELECT COUNT("IATA_CODE"), COUNT("AIRPORT"), COUNT("CITY"), 
                COUNT("STATE"), COUNT("COUNTRY"), COUNT("LATITUDE"), 
                COUNT("LONGITUDE")
        FROM (
            SELECT "IATA_CODE", "AIRPORT", "CITY", "STATE", 
                    "COUNTRY", "LATITUDE", "LONGITUDE"
            FROM (
                SELECT * FROM airports
            ) VERTICAPY_SUBTABLE
        ) VERTICAPY_SUBTABLE 
        LIMIT 1', 
        rowset=500) OVER ()


Let's also look at the :py:func:`~verticapy.vDataFrame.min` method:

.. code-block:: python

    # Finding minimum in the ID column of Ext_Table
    Ext_Table["LATITUDE"].min()

.. code-block:: sql

    SELECT
        DBLINK(USING PARAMETERS cid='pgdb', query='
            SELECT MIN("LATITUDE")
            FROM (
                SELECT "IATA_CODE", "AIRPORT", "CITY", "STATE", 
                       "COUNTRY", "LATITUDE", "LONGITUDE"
                FROM (
                    SELECT * FROM airports
                ) VERTICAPY_SUBTABLE
            ) VERTICAPY_SUBTABLE 
            LIMIT 1', 
            rowset=500) OVER ()

For the above examples, the queries were pushed to the external database.

If the function is unique to Vertica, it automatically fetches the data from the external database to compute on the Vertica server.

Let's try an example with the :py:func:`~verticapy.vDataFrame.describe` function, which is a unique Vertica function.    

.. code-block:: python

    # Describe the main attributes of numerical columns in the Ext_table
    Ext_Table.describe()

.. code-block:: sql

    -- Getting the version
    SELECT
        /*+LABEL('utilities.version')*/ version();

    -- Computing the descriptive statistics of all numerical columns using SUMMARIZE_NUMCOL
    SELECT
        /*+LABEL('vDataFrame.describe')*/ SUMMARIZE_NUMCOL("LATITUDE", "LONGITUDE") OVER () 
    FROM (
        SELECT
            "IATA_CODE",
            "AIRPORT",
            "CITY",
            "STATE",
            "COUNTRY",
            "LATITUDE",
            "LONGITUDE"
        FROM (
            SELECT
                DBLINK(USING PARAMETERS cid='pgdb', query='
                    SELECT * 
                    FROM airports', rowset=500) OVER ()
        ) VERTICAPY_SUBTABLE
    ) VERTICAPY_SUBTABLE;

.. ipython:: python
    :suppress:
    :okwarning:

    res = tab_data.describe()
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_describe.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_describe.html

We can see that the data was fetched from the external database to be computed in Vertica.

Now we can turn off SQL display.

.. code-block:: python

    # Turning off SQL display
    vp.set_option("sql_on", False)

Using SQL Magic Cells
---------------------

.. ipython:: python

    # Load extension for running SQL magic cells
    %load_ext verticapy.sql

We can use magic cells to call external tables using special characters 
like ``$$$`` and ``%%%``. If we have multiple external databases, we can specify special characters for each.

This makes writing queries a lot more convenient and visually appealing!

Now we will try to get fetch data from our external database ``pgdb``, whose special character is ``&``.

.. code-block:: python

    %%sql
    /* Getting all data from airports table which is placed in the PostgreSQL database represented by "&". */
    SELECT * FROM &&& airports &&&;

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_table.html

To perform all regular queries, all we need to do is call the table with its name inside three special characters.

We'll now try out some queries:

Count the elements inside the table:

.. code-block:: python

    %%sql
    /* Counting all elements inside the airports table in PostgreSQL. */
    SELECT COUNT(*) FROM &&& airports &&&;

.. ipython:: python
    :suppress:

    query = """
    SELECT COUNT(*) FROM public.airports
    """
    res = vp.vDataFrame(query)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_count.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_count.html

Find the ``IATA_CODE`` where ``CITY`` is ``Allentown``:

.. code-block:: python

    %%sql
    /* Finding IATA_CODE where the CITY is "Allentown" in the airports table. */
    SELECT IATA_CODE
    FROM &&& airports &&&
    WHERE CITY='Allentown';

.. ipython:: python
    :suppress:

    query = """
    SELECT IATA_CODE
    FROM  public.airports
    WHERE CITY='Allentown';
    """
    res = vp.vDataFrame(query)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_count_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_count_2.html

.. note:: Any query that we write inside the ``&&&`` signs is also sent to the external database to be run.


So, instead of just calling the whole table, we can query it using the same special character padding.

For example, let's select all elements inside the ``airports`` table:

.. code-block:: python

    %%sql
    /* Getting all data from airports table which is placed in the PostgreSQL database represented by "$". */
    &&& SELECT * FROM airports &&&;

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_table.html

Now we'll run a search query to find a particular id:

.. code-block:: python

    %%sql
    /* Finding IATA_CODE where the CITY is "Allentown" in the airports table. */
    &&& SELECT "IATA_CODE" FROM airports WHERE "CITY"='Allentown' &&&;

.. ipython:: python
    :suppress:

    query = """
    SELECT "IATA_CODE" FROM airports WHERE "CITY"='Allentown'
    """
    res = vp.vDataFrame(query)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_find.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_find.html

We can also ``insert`` a new entry into the airports table, 
which is placed in the postgreSQL database represented by ``&``:


.. code-block:: python

    %%sql
    /* Inserting an entry into the airports table which is placed in the postgreSQL database represented by "&". */
    &&& 
    INSERT INTO airports 
        ("IATA_CODE", "AIRPORT",        "CITY",    "STATE", "COUNTRY", "LATITUDE", "LONGITUDE") 
    VALUES ('MXX'      , 'Midway Airport', 'Chicago', 'IL',    'USA',     66.60,      35.00); 
    &&&

Connect Multiple Databases
---------------------------

You can connect and use multiple datasets from different databases.

In this example we will get:

- Airline data from ``PostgreSQL``
- Airport data from ``MySQL``
- Flights data from Vertica

The datasets can be found `here <https://www.kaggle.com/datasets/usdot/flight-delays>`_.

Airline Data in PostgreSQL
+++++++++++++++++++++++++++

We can set up a new connection in just one line by referencing the alias inside the connection files. As before, we will provide the special character symbol that is used to invoke the connection.

.. code-block:: python

    # Setting up a connection with a database given an alias "pgdb"
    vp.set_external_connection(
        cid="pgdb",
        rowset=500,
        symbol="$",
    )

Let's look at the airline table that we have in our ``postgreSQL`` database.

.. code-block:: python

    %%sql
    /* Fetch all the data from the table airports in "pgdb" database. */
    SELECT * FROM $$$ airline $$$;

.. ipython:: python
    :suppress:

    import verticapy as vp
    vp.drop("public.airline")
    tab_data = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/flights/airlines.csv",
        schema = "public",
        table_name = "airline")
    res = tab_data
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_airlines_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_airlines_table.html

Airports Data in MySQL
++++++++++++++++++++++

We can create another new connection by providing the ``cid`` reference for our ``MySQL`` database. We'll also provide a unique special character, which is not used for any other connection.

.. code-block:: python

    # Setting up a connection with a database given an alias "mysql"
    vp.set_external_connection(
        cid="mysql",
        rowset=500,
        symbol="&",
    )

Let's take a look at the airports table that we have in our ``MySQL`` database.

.. code-block:: python

    %%sql
    /* Fetch all the data from the table airports in "mysql" database */
    SELECT * FROM &&& airports &&&;

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_airports_table.html

Flights Data Vertica
+++++++++++++++++++++

We'll now read a locally stored ``CSV`` file with the flights data and materialize it in Vertica.

.. code-block:: python

    # Reading a csv file and naming the table flights_vertica
    flight_vertica = vp.read_csv(
        'flights.csv',
        table_name = "flight_vertica"
    )

.. code-block:: python

    %%sql
    /* Fetch all the data from the table flight_vertica. */
    SELECT * FROM flight_vertica;

.. ipython:: python
    :suppress:

    import verticapy as vp
    vp.drop("public.flight_vertica")
    tab_data = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/flights/flights.csv",
        schema = "public",
        table_name = "flight_vertica")
    res = tab_data
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_flights_table.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_flights_table.html

Joins and Queries Across Multiple Databases
--------------------------------------------

Now we can run queries that execute through multiple sources.

Let's try to find the ``TAIL_NUMBER`` and ``Departing City`` for all the flights by joining the two tables:

- ``flight_vertica`` (stored in Vertica)
- ``airports`` (stored in ``MySQL``)

.. code-block:: python

    %%sql
    /* Fetch TAIL_NUMBER and CITY after Joining the flight_vertica table with airports table in MySQL database. */
    SELECT flight_vertica.TAIL_NUMBER, airports.CITY AS Departing_City
    FROM flight_vertica
    INNER JOIN &&& airports &&&
    ON flight_vertica.ORIGIN_AIRPORT = airports.IATA_CODE;

.. ipython:: python
    :suppress:

    query = """
    SELECT flight_vertica.TAIL_NUMBER, public.airports.CITY AS Departing_City
    FROM public.flight_vertica
    INNER JOIN public.airports
    ON flight_vertica.ORIGIN_AIRPORT = public.airports.IATA_CODE;
    """
    res = vp.vDataFrame(query)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_multi_join.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_multi_join.html

Let's try another query to find the ``TAIL_NUMBER`` and ``AIRLINE`` of all the flights by joining the two tables:

- ``flight_vertica`` (stored in Vertica)
- ``airline`` (stored in ``PostgreSQL``)

.. code-block:: python

    %%sql
    /* Fetch TAIL_NUMBER and AIRLINE after Joining the flight_vertica table with airline table in PostgreSQL database. */
    SELECT flight_vertica.TAIL_NUMBER, airline.AIRLINE
    FROM flight_vertica
    INNER JOIN $$$ airline $$$ 
    ON flight_vertica.AIRLINE = airline.IATA_CODE;

.. ipython:: python
    :suppress:

    query = """
    SELECT public.flight_vertica.TAIL_NUMBER, public.airline.AIRLINE
    FROM public.flight_vertica
    INNER JOIN public.airline
    ON public.flight_vertica.AIRLINE = public.airline.IATA_CODE;
    """
    res = vp.vDataFrame(query)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_multi_join_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_multi_join_2.html

We can even try queries that require multiple joins.

In the following example, we try to get the ``TAIL_NUMBER``, 
``AIRLINE``, and ``CITY`` details for all the flights by joining:

- ``flight_local`` table (stored in Vertica)
- ``airline`` table (stored in ``PostgreSQL``)
- ``airports`` table (stored in ``MySQL``)

.. code-block:: python

    %%sql
    /* Fetch FLIGHT_NUMBER, AIRLINE and STATE after Joining the flight_vertica table with two other tables from different databases. */
    SELECT flight_vertica.FLIGHT_NUMBER, airline.AIRLINE, airports.STATE
    FROM flight_vertica
    INNER JOIN $$$ airline $$$ 
    ON flight_vertica.AIRLINE = airline.IATA_CODE
    INNER JOIN &&& airports &&&
    ON flight_vertica.ORIGIN_AIRPORT = airports.IATA_CODE;

.. ipython:: python
    :suppress:

    query = """
    SELECT flight_vertica.FLIGHT_NUMBER, airline.AIRLINE, airports.STATE
    FROM flight_vertica
    INNER JOIN airline
    ON flight_vertica.AIRLINE = airline.IATA_CODE
    INNER JOIN airports
    ON flight_vertica.ORIGIN_AIRPORT = airports.IATA_CODE;
    """
    res = vp.vDataFrame(query)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_multi_join_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_multi_join_2.html

Pandas.DataFrame
-----------------

The joins also work with ``pandas.Dataframe``. We can perform the same query that required multiple joins, but now with a local Pandas dataframe.

We can read a local passengers CSV file using :py:func:`~verticapy.read_csv` or we could create an artificial dataset as well.

.. code-block:: python

    # Create a Pandas Data Frame after importing the csv file "passengers.csv"
    import pandas as pd
    passengers_pandas = pd.read_csv('passengers.csv')

.. ipython:: python

    import numpy as np
    import pandas as pd

    # Set the parameters
    total_flights = 4000
    total_entries = 12000

    # Generate random flight numbers (with duplicates)
    flight_numbers = np.random.randint(1, total_flights + 1, total_entries)

    # Generate random passenger counts (1 to 300 passengers)
    passenger_counts = np.random.randint(1, 301, total_entries)

    # Create the DataFrame
    passengers_pandas = pd.DataFrame({
        'FLIGHT_NUMBER': flight_numbers,
        'PASSENGER_COUNT': passenger_counts
    })
    passengers_pandas

.. ipython:: python
    :suppress:
    :okwarning:

    vp.drop("public.passengers_pandas")
    passengers_pandas.to_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/flights/temp.csv",
        index=False
        )
    passengers_pandas = vp.read_csv(
        "SPHINX_DIRECTORY/source/_static/website/examples/data/flights/temp.csv",
        schema = "public",
        table_name = "passengers_pandas")

We can now perform the same query involving the three tables:

- ``flight_vertica`` table (stored in Vertica)
- ``passengers_pandas`` table (``pandas.DataFrame`` stored in-memory)
- ``airline`` table (stored in ``PostgreSQL``)
- ``airports`` table (stored in ``MySQL``)

.. code-block:: python

    %%sql
    SELECT 
        flight_vertica.TAIL_NUMBER, 
        airline.AIRLINE, 
        airports.CITY, 
        :passengers_pandas.PASSENGER_COUNT
    FROM flight_vertica
    INNER JOIN $$$ airline $$$ 
    ON flight_vertica.AIRLINE = airline.IATA_CODE
    INNER JOIN &&& airports &&&
    ON flight_vertica.ORIGIN_AIRPORT = airports.IATA_CODE
    INNER JOIN :passengers_pandas
    ON flight_vertica.FLIGHT_NUMBER = :passengers_pandas.FLIGHT_NUMBER;

.. ipython:: python
    :suppress:
    :okwarning:

    query = """
    SELECT 
        flight_vertica.TAIL_NUMBER, 
        airline.AIRLINE, 
        airports.CITY, 
        passengers_pandas.PASSENGER_COUNT
    FROM flight_vertica
    INNER JOIN public.airline 
    ON flight_vertica.AIRLINE = airline.IATA_CODE
    INNER JOIN public.airports
    ON flight_vertica.ORIGIN_AIRPORT = airports.IATA_CODE
    INNER JOIN passengers_pandas
    ON flight_vertica.FLIGHT_NUMBER = passengers_pandas.FLIGHT_NUMBER;
    """
    res = vp.vDataFrame(query)
    html_file = open("SPHINX_DIRECTORY/figures/ug_fs_dblink_multi_mega_join.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: SPHINX_DIRECTORY/figures/ug_fs_dblink_multi_mega_join.html

Conclusion
-----------

With the combination of VerticaPy and ``DBLINK``, we can now work with multiple datasets stored in different databases. We can work simultaneously with external tables, Vertica tables, and Pandas DataFrame in a **single query**! There is no need to materialize the table before use because it's all taken care of in the background.

The cherry on the cake is the ease-of-use that is enabled by VerticaPy and its Python-like syntax.

Queries that required paragraph upon paragraph to execute can now be done **efficiently** with only a **few intuitive lines of code**.

This new functionality opens up many possibilities for data querying and manipulation in Vertica.