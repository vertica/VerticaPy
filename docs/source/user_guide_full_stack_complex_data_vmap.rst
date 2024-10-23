.. _user_guide.full_stack.complex_data_vmap:

==========================
Complex Data Types & VMaps
==========================


Setup
------


In order to work with complex 
data types in VerticaPy, you'll need to 
complete the following three setup tasks:

- Import relevant libraries:

.. code-block:: python

    import verticapy as vp

- Connect to Vertica:

.. code-block:: python

    vp.new_connection(
        {
            "host": "10.211.55.14", 
            "port": "5433", 
            "database": "testdb", 
            "password": "XxX", 
            "user": "dbadmin"
        },
        name = "Vertica_New_Connection"
    )


Check your VerticaPy version to make sure you have access to the right functions:

.. ipython:: python

    @suppress
    import verticapy as vp
    vp.__version__

You can make it easier to keep track of your work by creating a custom schema:

.. note:: Because some tables are repeated in this demonstration, tables with the same names are dropped.

.. ipython:: python

    vp.drop("complex_vmap_test", method = "schema")
    vp.create_schema("complex_vmap_test")

We also set the path to our data:

.. code-block:: python

    path= "/home/dbadmin/"


You can download the demo datasets from `here <https://github.com/vertica/VerticaPy/tree/master/verticapy/datasets/data>`_.


Loading Complex Data
---------------------

There are two ways to load a nested data file:

- Load directly using :py:func:`verticapy.read_json`. 
    In this case, you will need to use an additional parameter 
    to identify all the data types. The function loads the 
    data using flex tables and VMaps (Native Vertica MAPS, 
    which are flexible but not optimally performant).

- Load using :py:func:`verticapy.read_file`. The function preidcts the complex data structure.


Let's try both:

.. code-block:: python

    import verticapy as vp
    data = vp.read_json(
        path + "laliga/2008.json",
        schema = "public",
        ingest_local = False,
        use_complex_dt = True,
        genSQL = True
    )

Similar to the use of :py:func:`verticapy.read_json` above, 
we can use :py:func:`verticapy.read_file` to ingest the complex data directly:

.. code-block:: python

    data = vp.read_file(
        path = path + "laliga/2005.json",
        ingest_local = False,
        schema = "complex_vmap_test",
    )
    data

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_laliga
    data = load_laliga()
    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_data.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_data.html

We can also use the handy ``genSQL`` parameter to generate 
(but not execute) the SQL needed to create the final relation:

.. note:: This is a great way to customize the data ingestion or alter the final relation types.

.. code-block:: SQL

    CREATE TABLE "complex_vmap_test"."laliga_2005" (
        "away_score" FLOAT,
        "away_team" ROW(
            "away_team_gender" VARCHAR(60),
            "away_team_group" VARCHAR(60),
            "away_team_id" INT,
            "away_team_name" VARCHAR(60),
            "country" ROW(
                "id" INT,
                "name" VARCHAR(60)
            )
        ),
        "competition" ROW(
            "competition_id" INT,
            "competition_name" VARCHAR(60),
            "country_name" VARCHAR(60)
        ),
        "competition_stage" ROW(
            "id" INT,
            "name" VARCHAR(60)
        ),
        "home_score" INT,
        "home_team" ROW(
            "country" ROW(
                "id" INT,
                "name" VARCHAR(60)
            ),
            "home_team_gender" VARCHAR(60),
            "home_team_group" VARCHAR(60),
            "home_team_id" INT,
            "home_team_name" VARCHAR(60)
        ),
        "kick_off" TIME,
        "last_updated" DATE,
        "match_date" DATE,
        "match_id" INT,
        "match_status" VARCHAR(60),
        "match_week" INT,
        "metadata" ROW(
            "data_version" DATE,
            "shot_fidelity_version" INT,
            "xy_fidelity_version" INT
        ),
        "season" ROW(
            "season_id" INT,
            "season_name" VARCHAR(60)
        )
    );

    COPY "complex_vmap_test"."laliga_2005"
    FROM '/scratch_b/qa/ericsson/laliga/2005.json'
    PARSER FJsonParser();



Feature Exploration
---------------------


In the generated SQL from the above example, we can see 
that the ``away_team`` column is a ROW type with a complex 
structure consisting of many sub-columns. We can convert 
this column into a JSON and view its contents:

.. code-block:: python

    data["competition_stage"].astype("json")

.. ipython:: python
    :suppress:

    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_data_2.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_data_2.html

As with a normal vDataFrame, we can easily extract the values from the sub-columns:

.. code-block:: python

    data["away_team"]["away_team_gender"]

.. ipython:: python
    :suppress:

    res = data["away_team"]["away_team_gender"]
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_data_3.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_data_3.html

We can view any nested data structure by index:

.. code-block:: python

    ddata["competition"]["competition_id"]

.. ipython:: python
    :suppress:

    res = data["competition"]["competition_id"]
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_nested.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_nested.html

These nested structures can be used to create features:

.. ipython:: python

    data["name_home"] = data["home_team"]["home_team_name"];


We can even flatten the nested structure inside a json file, 
either flattening the entire file or just particular columns:

.. code-block:: python

    data = vp.read_json(
        path = path + "laliga/2008.json",
        table_name = "laliga_flat",
        schema = "complex_vmap_test",
        ingest_local = False,
        flatten_maps = True,
    )
    data

.. ipython:: python
    :suppress:

    vp.drop("complex_vmap_test.laliga_flat")
    path = "/project/data/VerticaPy/docs"
    path = path[0:-5] + "/verticapy/datasets/data/"
    data = vp.read_json(path = path + "laliga/2008.json",
                        table_name = "laliga_flat",
                        schema = "complex_vmap_test",
                        ingest_local = True,
                        flatten_maps=True,)
    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_flatten.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_flatten.html

We can see that all the columns from the JSON file have been 
flattened and multiple columns have been created for each 
sub-column. This causes some loss in data structure, 
but makes it easy to see the data and to use it for model building.

It is important to note that the data type of certain 
columns (home_team.managers) is now ``VMap``, and not the ``ROW`` 
type that we saw in the above cells. Even though both are 
used to capture nested data, there is in a subtle difference between the two.

**VMap:** More flexible as it stores the data as a string of maps, 
allowing the ingestion of data in varying shapes. The shape is 
not fixed and new keys can easily be handled. This is a great 
option when we don't know the structure in advance, or if the structure changes over time.

**Row:** More rigid because the dictionaries, including 
all the data types, are fixed when they are defined. Newly 
parsed keys are ignored. But because of it's rigid structure, 
it is much more performant than VMaps. They are best used when 
the file structure is known in advance.


To deconvolve the nested structure, we can use the ``flatten_arrays``
parameter in order to make the output strictly formatted. However, it 
can be an expensive process.

.. code-block:: python

    vp.drop("complex_vmap_test.laliga_flat")
    data = vp.read_json(path = path + "laliga/2008.json",
                        table_name = "laliga_flat",
                        schema = "complex_vmap_test",
                        ingest_local = False,
                        flatten_arrays=True,)
    data

.. ipython:: python
    :suppress:

    vp.drop("complex_vmap_test.laliga_flat")
    path = "/project/data/VerticaPy/docs"
    path = path[0:-5] + "/verticapy/datasets/data/"
    data = vp.read_json(path = path + "laliga/2008.json",
                        table_name = "laliga_flat",
                        schema = "complex_vmap_test",
                        ingest_local = True,
                        flatten_arrays=True,)
    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_flatten_arrays.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_flatten_arrays.html

We can even convert columns into other formats, such as string:

.. code-block:: python

    data["home_team.managers.0.nickname"].astype(str)

.. ipython:: python
    :suppress:

    data["home_team.managers.0.nickname"].astype(str)
    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_flatten_arrays_astype.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_flatten_arrays_astype.html

Or integer:


.. code-block:: python

    data["match_week"].astype(int)

.. ipython:: python
    :suppress:

    data["match_week"].astype(int)
    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_flatten_arrays_astype_int.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_flatten_arrays_astype_int.html

It is also possible to:

- Cast ``str`` to ``array``
- Cast complex data types to ``json`` str
- Cast ``str`` to ``VMAP``s
- And much more...

Multiple File Ingestion
------------------------

If we have multiple files with the same extension, we can easily ingest them using the "*" operator:

.. code-block:: python

    data = vp.read_file(
        path = path + "laliga/*.json",
        table_name = "laliga_all",
        ingest_local = False,
        schema = "complex_vmap_test",
    )

We can also do this for other file types. For example, csv:

.. code-block:: python

    data = vp.read_csv(
        path = path + "*.csv",
        table_name = "cities_all",
        schema = "complex_vmap_test",
        ingest_local = False,
        insert = True
    )

Materialize
------------

When we do not materialize a table, it automatically becomes a flextable:

.. code-block:: python

    data = vp.read_json(
        path = path + "laliga/*.json",
        table_name = "laliga_verticapy_test_json",
        schema = "complex_vmap_test",
        ingest_local = False,
        materialize = False,
    )

.. ipython:: python
    :suppress:

    vp.drop("complex_vmap_test.laliga_verticapy_test_json")
    path = "/project/data/VerticaPy/docs"
    path = path[0:-5] + "/verticapy/datasets/data/"
    data = vp.read_json(path = path + "laliga/*.json",
                        table_name = "laliga_verticapy_test_json",
                        schema = "complex_vmap_test",
                        ingest_local = True,
                        materialize = False,)
    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_materialize.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_materialize.html

Some of the columns are VMAPs:

.. ipython:: python

    managers = ["away_team.managers", "home_team.managers"]
    for m in managers:
        print(data[m].isvmap())

We can easily flatten the VMaps virtual columns by 
using the :py:func:`vDataFrame.flat_vmap` method:

.. code-block:: python

    data.flat_vmap(managers).drop(managers)


.. ipython:: python
    :suppress:

    data.flat_vmap(managers).drop(managers)
    res = data
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_materialize_flat.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. .. raw:: html
..     :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_materialize_flat.html

To check for a flex table, we can use the following function:

.. ipython:: python
    
    from verticapy.sql import isflextable

    isflextable(table_name = "laliga_verticapy_test_json", schema = "complex_vmap_test")

We can then manually materialize the flextable using the 
convenient :py:func:`vDataFrame.to_db` method:

.. ipython:: python

    @suppress
    vp.drop("complex_vmap_test.laliga_to_db")
    data.to_db("complex_vmap_test.laliga_to_db");

Once we have stored the database, we can easily create 
a :py:func:`vDataFrame` of the relation:


.. ipython:: python

    data_new = vp.vDataFrame("complex_vmap_test.laliga_to_db")


Transformations
-----------------

First, we load the dataset.

.. code-block:: python

    from verticapy.datasets import load_amazon
    
    data = load_amazon()

.. ipython:: python
    :suppress:

    from verticapy.datasets import load_amazon
    res = data = load_amazon()
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_cities.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_cities.html

Once we have data in the form of :py:func:`vDataFrame`, 
we can readily convert it to a ``JSON`` file:

.. ipython:: python

    data.to_json(path = "amazon_json.json");

Now we can load the new JSON file and see the contents:

.. code-block:: python

    data = read_json(
        path = "amazon_json.json",
        schema = "complex_vmap_test",
        table_name = "cities_transf_test",
        ingest_local = False,
    )

We can even extract the ``JSON`` as string and edit it before saving it as a json file:

.. ipython:: python

    json_str = data.to_json();

Let's look at the begining portion of the string:

.. ipython:: python

    json_str[0:100]

We can edit a portion of the string and save it again. 
We'll change the name of the first State from ACRE to XXXX:   

.. ipython:: python

    json_str = json_str[:35] + 'XXXX' + json_str[39:];

Now we can save this edited strings file:

.. ipython:: python


    out_file = open(path + "amazon_edited.json", "w")
    out_file.write(json_str)
    out_file.close()

If we look at the new file, we can see the updated changes:

.. ipython:: python

    vp.drop("complex_vmap_test.amazon_edit")
    data = vp.read_json(
        path = path + "amazon_edited.json",
        schema = "complex_vmap_test",
        table_name = "amazon_edit",
        ingest_local = True,
    );

Let's search for the changed name:

.. code-block:: python

    data[data["state"] == "XXXX"]

.. ipython:: python
    :suppress:

    res = data[data["state"] == "XXXX"]
    html_file = open("/project/data/VerticaPy/docs/figures/ug_fs_complex_cities_search.html", "w")
    html_file.write(res._repr_html_())
    html_file.close()

.. raw:: html
    :file: /project/data/VerticaPy/docs/figures/ug_fs_complex_cities_search.html

Now to clean everything up, we can drop our temporary schema:

.. ipython:: python

    vp.drop("complex_vmap_test", method = "schema")

Conclusion
-----------

This new functionality not only make it easy to ingest complex data types in different formats, but it enables data wrangling like never before.

The new features provide increased flexibility while keeping the process and syntax simple. You can do all of the following in VerticaPy:

- Ingest complex datasets.
- Perform convenient column operations.
- Switch data types.
- Flatten columns and maps into array like structures.