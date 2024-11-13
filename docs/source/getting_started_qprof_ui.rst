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

.. image:: image dir...

The first you tab you notice here is the Query Plan Tree tab. It displays the query plan in a graphical tree format with color coding for different metrics of the query e.g. time, memory, cost, etc. There are different ettings that you can use to explore the query plan. They are listed below:

1. Metrics

Here you can change the metrics that you want to see by clicking on the ``Metrics`` tab. You can also select the tooltips that you want to see by checking the information that you want to be displayed.


2. Path ID

In this tab you can filter the path ids that you are interested in. You can do this directly be selecting the particular ``path id`` from the dropdown menu. You can also search by tooltip by entering the part of tooltip that you want to match. This could be name of the table that you notice in the tooltip. You can also search by filtering operators. Use the dropdowns to select either one or two opeartors. For example, criterea 1 could be ``JOIN`` and criterea 2 could be ``SCAN``. To go back to the orginal tree, you can click on the ``Reset`` button.

3. Tree Style

The first option here is to choose between displaying ``Temporary Relations`` combined or separate. The next option is to toggle the display of ``DML Projections``

4. Query Text

Here you will see the query text of the query that you are currently looking at. 

5. Session PARAMETERS

All the ``SESSION PARAMETERS`` that are non-default will be listed here.

6. Detailed tooltip

If you want to look deeper into the tooltip of a sepcific path id, you can select it from the dropdown menu.

7. Summary

The key elements of the query execution plan are listed here.

8. Navigation buttons

If you have multiple queries in your profile data, then you can use the navigation buttons to switch between the queries.


Query Events Tab
-----------------

.. image:: image dir...

The ``Query Events`` tab displays the query events for the query that you are currently looking at.

You can select the query that you want to look it from the dropdown menu, and then click ``Run Statistics`` to get the statistics for the query.

.. note:: You can head  over to the ``Transactions`` tab to get the exact query index of the query you want to run statistics for.

On the top it will display a summary in the form of a half-pie chart indicating the number of alerts for each category: ``OPTIMIZER``, ``NETWORK``, and ``EXECUTION``.

Below, all the events will be displayed. You can filter with ``Node``, ``Category``, and ``Type`` to find particular events. 

Lastly, you can click on the ``Take Action`` button to resolve the action automatically.

.. warning:: This can have serious consequences on your data. So double the SQL that is about to be run.

Query Engines Tab
-----------------

.. image:: image dir...

In this tab, we display all the data from the ``QueryEngineExection`` table. We have two main types of display:

1. Summary

Here all the aggregates are displayed either node-wise or path_id-wise.

2. Detailed

Here the users have the option to look at either the raw table or a pivot table. For the pivot table, the users can select the specific metrics they want to see from the checklist on the right side.

Transactions Tab
----------------

.. image:: image dir...

In this tab, we display all the important information from the ``QueryRequest`` table.

Plots Tab
---------

.. image:: image dir...

.. warning:: This tab is still under development and may result in some issues. 


Other Info
----------

.. image:: image dir...

In this tab, we have two tables that display the ``Slow Events`` and ``Optimizer Events``. But we filter the information only for the specific query that is profiled.

Manual Query
------------

.. image:: image dir...

We allow the users to run SQL directly from this tab. This is useful if the user wants to run a query that is not part of the profile data. 

Explain Plan
------------

.. image:: image dir...

We allow the users to run ``EXPLAIN PLAN`` directly from this tab. To get the desired results, you must be connected to your database that you ran the query on.

Advanced vDataFrame Output settings
-------------------------------------

.. image:: image dir...

The users can modify the ``vDataFrame`` display settings as per their desire. This mean that the users can change the ``display.max_rows`` or removing/inserting commas for numerical data.


Consistency Checks
------------------

.. image:: image dir...

Here we display the results of the consistency checks that we have run on the profile data. This tells us if some information in the tables were removed because of rentention issues. 

Execution/Error Details
------------------------

All the execution details and error details of the queries are displayed here. This is useful if you want to see the details of the execution and error of the query that you are currently looking at.

