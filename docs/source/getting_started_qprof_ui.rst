.. _verticapylab_gs.queryprofiler:

=============================
Query Profiler User Interface
=============================

Query Profiler provides a user-friendly way to analyze your queries and find improvement opportunities.
Whether you are an expert or a beginner, it can offer you some good inisghts into your query.

To use this, you need:

1. VerticaPyLab
2. Vertica database

To install and setup (1), please follow the instruction here: :ref:`verticapylab_gs`. This page also has instruction for the cases where you do not have a Vertica database and would like to try out the Vertica CE.

Now let us finally explain what the different UI functionalities are to this date i.e. 11/13/2024.

.. note:: The UI has been transforming based on user-feedback so some elements may have been moved around.

.. _main_page:

VerticaPyLab Main Page
======================

When you start up the VerticaPyLab, you will be ending up on the launcher page. Here you can use to connect button to connect to your databse, or if you are connected, then directly click on the Query Profiler button.

.. image:: image dir...

.. _qprof_main_page:

Query Profiler Page
===================

Once you click on the Query Profiler button, you will be directed to the Query Profiler page. 

The header of this page has the following elements:

- The home button: This will take you back to the main page: :ref:`main_page`.
- The connect button: This will take you to the connect page.


If you are connected to the database, you should be able to see your database and username under the connection details.

.. image:: image dir...


.. note:: If you need to learn more about the QueryProfiler tool, then you can click on the link at the right corner.


You can use this page in three different ways:

1. To load an existing profile data that has been stored in a ``target schema`` with a specific ``key``.
2. To run and profile a new query (or a list of queries) and, optionally, store it in a ``target schema`` with a specific ``key``.
3. Compare two different queries by either retrieving the existing profile data stored in a ``target schema`` with a specific ``key`` or by running and profiling a new query.

Let's look at each of these use cases in the following sections.

Load Existing Profile Data
---------------------------

In order to retrieve a saved profile data, you need to provide the ``key`` and ``target schema``. You have the option to enter it manually or use the dropdowns provided. Note that the ``target_schema`` you provide should already exist. 

Once you have provided these, you can click on the ``Get Query Plan`` button to load the profile data into the UI.

.. image:: image dir...

Additionally, there is an option to load the data from a ``tar file``. You can click on the ``From a file`` tabe and select the tar file. Don't forget to confirm the selection. 

Once you have provided these, you can click on the ``Get Query Plan`` button to load the profile data into the UI.

.. image:: image dir...

Creating a new Query Profile
----------------------------

You can create a new query profile by using two methods:

1. By using the ``transaction_id`` and ``statement_id`` of the query. This means that the query should already have been run.

1. By using the SQL of the query. 

.. note:: In all the options, the ``target_schema`` and ``key`` are optional. If you leave them empty then a temporary schema and key will be used which you may not be able to retrieve later.

You could search for a query using the search tab. Then you can extract the ``transaction_id`` and ``statement_id`` of the query and use them to create a new query profile. For the search you have the option to search using the custom label that the user has provided or the query text itself. You can search for exact match or partial match.

.. image:: image dir...

To profile a query with transaction id and statement id, just enter the values in the provided text boxes and click on the ``Get Query Plan`` button.

.. image:: image dir...

To profile the query using SQL, you can click the ``From SQL`` tab. In this tab, you can enter the SQL of the query and click on the ``Get Query Plan`` button. You can even enter multiple queries by separating them with a semicolon. For example:

.. code-block:: sql

  select * from table1; select * from table2;

The last tab in this section is the ``Multiple Queries`` tab. Here you can enter a list of tuples of ``transaction_id`` and ``statement_id``. For example:

.. code-block:: Python

  [(12345678,1), (23456789,1)]

Once you have provided these, you can click on the ``Get Query Plan`` button to load the profile data into the UI. Remember you can optionally provide the ``target_schema`` and ``key`` to store the profile data.

.. image:: image dir...

Compare Queries
---------------

You can provide information about two queries to compare them. This is done by either retrieving the existing profile data stored in a ``target schema`` with a specific ``key`` or by running and profiling a new query.

To retrieve the existing profile data stored in a ``target schema`` with a specific ``key``, just enter the ``target_schema`` and ``key`` and click on the ``Compare`` button.

To profile an unsaved query, you can either enter the ``transaction_id`` and ``statement_id`` of the query or use the SQL of the query.

Lastly, you have the optional to change ``SESSION PARAMETERS`` for your query. All you need to do is enter the session control SQL and click on the ``Compare`` button. for example:

.. code-block:: sql

  ALTER SESSION SET statement_mem = 100000000;


.. image:: image dir...


Query Plan Tree Page
=====================

Once you have loaded the profile data, you can see the query plan in the Query Plan Tree page.


.. image:: image dir...


Let's go over the different elements of the Query Plan Tree page:

Download tab
------------

The ``Download`` tab. If you click the ``Save qprof Information & Report`` button, it will download the profile data as a tar file. And also save the profile report as an HTML file that can be viewed offline.

Query Plan Tree
---------------

The first you tab you notice here is the Query Plan Tree tab. It displays the query plan in a graphical tree format with color coding for different metrics of the query e.g. time, memory, cost, etc. There are different ettings that you can use to explore the query plan. They are listed below:

1. Metrics

Here you can change the metrics that you want to see by clicking on the ``Metrics`` tab. You can also select the tooltips that you want to see by checking the information that you want to be displayed.


2. Path ID

In this tab you can filter the path ids that you are interested in. You can do this directly be selecting the particular ``path id`` from the dropdown menu. You can also search by tooltip by entering the part of tooltip that you want to match. This could be name of the table that you notice in the tooltip. You can also search by filtering operators. Use the dropdowns to select either one or two opeartors. For example, criterea 1 could be ``JOIN`` and criterea 2 could be ``SCAN``. To go back to the orginal tree, you can click on the ``Reset`` button.

3. Tree Style

The first option here is to choose between displaying ``Temporary Relations`` combined or separate. The next option is to toggle the display of ``DML Projections``
