"""
Copyright  (c)  2018-2024 Open Text  or  one  of its
affiliates.  Licensed  under  the   Apache  License,
Version 2.0 (the  "License"); You  may  not use this
file except in compliance with the License.

You may obtain a copy of the License at:
http://www.apache.org/licenses/LICENSE-2.0

Unless  required  by applicable  law or  agreed to in
writing, software  distributed  under the  License is
distributed on an  "AS IS" BASIS,  WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the  License for the specific  language governing
permissions and limitations under the License.
"""
import copy
import os
from pathlib import Path
import re
from typing import Any, Callable, Literal, Optional, Union, TYPE_CHECKING
import uuid
import warnings

from tqdm.auto import tqdm

from verticapy.errors import EmptyParameter, ExtensionError, QueryError

from verticapy.core.vdataframe import vDataFrame

import verticapy._config.config as conf
from verticapy._typing import NoneType, PlottingObject, SQLExpression
from verticapy._utils._parsers import parse_explain_graphviz
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, format_query, format_type
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import vertica_version

from verticapy.performance.vertica.collection.profile_export import ProfileExport
from verticapy.performance.vertica.collection.profile_import import ProfileImport
from verticapy.performance.vertica.qprof_utility import QprofUtility
from verticapy.performance.vertica.tree import PerformanceTree
from verticapy.plotting._utils import PlottingUtils
from verticapy.sql.dtypes import get_data_types

if TYPE_CHECKING and conf.get_import_success("graphviz"):
    from graphviz import Source


class QueryProfiler:
    """
    Base class to profile queries.

    .. important::

        This class is currently under development
        and remains in beta. Please note that the
        parameters may undergo changes in the future.
        We plan to introduce a stable version in
        VerticaPy 1.0.3 or later.

    .. important::

        Most of the classes are not available in
        Version 1.0.0. Please use Version 1.0.1
        or higher. Alternatively, you can use the
        ``help`` function to explore the functionalities
        of your current documentation.

    The :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
    is a valuable tool for anyone seeking to comprehend
    the reasons behind a query's lack of performance.
    It incorporates a set of functions inspired by the
    original QPROF project, while introducing an
    enhanced feature set. This includes the capability
    to generate graphics and dashboards, facilitating
    a comprehensive exploration of the data.

    Moreover, it offers greater convenience by
    allowing interaction with an object that
    encompasses various methods and expanded
    possibilities. To initiate the process, all
    that's required is a ``transaction_id`` and a
    ``statement_id``, or simply a query to execute.

    Parameters
    ----------
    transactions: str | tuple | list, optional
        Six options are possible for this parameter:

        - An ``integer``:
            It will represent the ``transaction_id``,
            the ``statement_id`` will be set to 1.
        - A ``tuple``:
            ``(transaction_id, statement_id)``.
        - A ``list`` of ``tuples``:
            ``(transaction_id, statement_id)``.
        - A ``list`` of ``integers``:
            the ``transaction_id``; the ``statement_id``
            will automatically be set to 1.
        - A ``str``:
            The query to execute.
        - A ``list`` of ``str``:
            The ``list`` of queries to execute. Each
            query will be execute iteratively.

            .. warning::

                It's important to exercise caution; if the
                query is time-consuming, it will require a
                significant amount of time to execute before
                proceeding to the next steps.

        .. note::

            A combination of the three first options can
            also be used in a ``list``.
    key_id: int, optional
        This parameter is utilized to load information
        from another ``target_schema``. It is considered
        a good practice to save the queries you intend
        to profile.
    resource_pool: str, optional
        Specify the name of the resource pool to utilize
        when executing the query. Refer to the Vertica
        documentation for a comprehensive list of available
        options.

        .. note::

            This parameter is used only when ``request`` is
            defined.
    target_schema: str | dict, optional
        Name of the schemas to use to store
        all the Vertica monitor and internal
        meta-tables. It can be a single
        schema or a ``dictionary`` of schema
        used to map all the Vertica DC tables.
        If the tables do not exist, VerticaPy
        will try to create them automatically.
    overwrite: bool, optional
        If set to ``True`` overwrites the
        existing performance tables.
    add_profile: bool, optional
        If set to ``True`` and the request does not include
        a profile, this option adds the profile keywords at
        the beginning of the query before executing it.

        .. note::

            This parameter is used only when ``request``
            is defined.
    check_tables: bool, optional
        If set to ``True`` all the transactions of
        the different Performance tables will be
        checked and a warning will be raised in
        case of incomplete data.

        .. warning::

            This parameter will aggregate on many
            tables using many parameters. It will
            make the process much more expensive.
    iterchecks: bool, optional
        If set to ``True``, the checks are done
        iteratively instead of using a unique
        SQL query. Usually checks are faster
        when this parameter is set to ``False``.

        .. note::

            This parameter is used only when
            ``check_tables is True``.

    Attributes
    ----------
    transactions: list
        ``list`` of ``tuples``:
        ``(transaction_id, statement_id)``.
        It includes all the transactions
        of the current schema.
    requests: list
        ``list`` of ``str``:
        Transactions Queries.
    request_labels: list
        ``list`` of ``str``:
        Queries Labels.
    qdurations: list
        ``list`` of ``int``:
        Queries Durations (seconds).
    key_id: int
        Unique ID used to build up the
        different Performance tables
        savings.
    request: str
        Current Query.
    qduration: int
        Current Query Duration (seconds).
    transaction_id: int
        Current Transaction ID.
    statement_id: int
        Current Statement ID.
    target_schema: dict
        Name of the schema used to store
        all the Vertica monitor and internal
        meta-tables.
    target_tables: dict
        Name of the tables used to store
        all the Vertica monitor and internal
        meta-tables.
    v_tables_dtypes: list
        Datatypes of all the performance
        tables.
    tables_dtypes: list
        Datatypes of all the loaded
        performance tables.
    overwrite: bool
        If set to ``True`` overwrites the
        existing performance tables.

    Examples
    --------

    Initialization
    ^^^^^^^^^^^^^^^

    First, let's import the
    :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
    object.

    .. ipython:: python
        :okwarning:

        from verticapy.performance.vertica import QueryProfiler

    There are multiple ways how we can use the
    :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.

    - From ``transaction_id`` and ``statement_id``
    - From SQL generated from verticapy functions
    - Directly from SQL query

    **Transaction ID and Statement ID**

    In this example, we run a groupby
    command on the amazon dataset.

    First, let us import the dataset:

    .. code-block:: python

        from verticapy.datasets import load_amazon

        amazon = load_amazon()

    Then run the command:

    .. code-block:: python

        query = amazon.groupby(
            columns = ["date"],
            expr = ["MONTH(date) AS month, AVG(number) AS avg_number"],
        )

    For every command that is run, a query is logged in
    the ``query_requests`` table. We can use this table to
    fetch the ``transaction_id`` and ``statement_id``.
    In order to access this table we can use SQL Magic.

    .. code-block:: python

        %load_ext verticapy.sql

    .. code-block:: python

        %%sql
        SELECT *
        FROM query_requests
        WHERE request LIKE '%avg_number%';

    .. hint::

        Above we use the ``WHERE`` command in order
        to filter only those results that match our
        query above. You can use these filters to
        sift through the list of queries.

    Once we have the ``transaction_id`` and
    ``statement_id`` we can directly use it:

    .. code-block:: python

        qprof = QueryProfiler((45035996273800581, 48))

    .. important::

        To save the different performance tables
        in a specific schema use
        ``target_schema='MYSCHEMA'``, 'MYSCHEMA'
        being the targetted schema. To overwrite
        the tables, use: ``overwrite=True``.
        Finally, if you just need local temporary
        table, use the ``v_temp_schema`` schema.

        Example:

        .. code-block:: python

            qprof = QueryProfiler(
                (45035996273800581, 48),
                target_schema='v_temp_schema',
                overwrite=True,
            )

    **Multiple Transactions ID and Statements ID**

    You can also construct an object based on multiple
    transactions and statement IDs by using a list of
    transactions and statements.

    .. code-block:: python

        qprof = QueryProfiler(
            [(tr1, st2), (tr2, st2), (tr3, st3)],
            target_schema='MYSCHEMA',
            overwrite=True,
        )

    A ``key_id`` will be generated, which you can
    then use to reload the object.

    .. code-block:: python

        qprof = QueryProfiler(
            key_id='MYKEY',
            target_schema='MYSCHEMA',
        )

    You can access all the transactions
    of a specific schema by utilizing
    the 'transactions' attribute.

    .. code-block:: python

        qprof.transactions

    **SQL generated from VerticaPy functions**

    In this example, we can use the Titanic dataset:

    .. code-block:: python

        from verticapy.datasets import load_titanic

        titanic= load_titanic()

    Let us run a simple command to get the average
    values of the two columns:

    .. code-block:: python

        titanic["age","fare"].mean()

    We can use the ``current_relation`` attribute to extract
    the generated SQL and this can be directly input to the
    Query Profiler:

    .. code-block:: python

        qprof = QueryProfiler(
            "SELECT * FROM " + titanic["age","fare"].fillna().current_relation()
        )

    **Directly From SQL Query**

    The last and most straight forward method is by
    directly inputting the SQL to the Query Profiler:

    .. ipython:: python
        :okwarning:

        qprof = QueryProfiler(
            "select transaction_id, statement_id, request, request_duration"
            " from query_requests where start_timestamp > now() - interval'1 hour'"
            " order by request_duration desc limit 10;"
        )

    The query is then executed, and you
    can easily retrieve the statement
    and transaction IDs.

    .. ipython:: python

        tid = qprof.transaction_id
        sid = qprof.statement_id
        print(f"tid={tid};sid={sid}")

    Or simply:

    .. ipython:: python

        print(qprof.transactions)

    To avoid recomputing a query, you
    can also directly use its statement
    ID and its transaction ID.

    .. ipython:: python
        :okwarning:

        qprof = QueryProfiler((tid, sid))

    Accessing the different Performance Tables
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We can easily look at any Vertica
    Performance Tables easily:

    .. code-block:: python

        qprof.get_table('dc_requests_issued')

    .. ipython:: python
        :suppress:
        :okwarning:

        result = qprof.get_table('dc_requests_issued')
        html_file = open("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_table_1.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_table_1.html

    .. note::

        You can use the method without parameters
        to obtain a list of all available tables.

        .. ipython:: python

            qprof.get_table()

    We can also look at all the
    object queries information:

    .. code-block:: python

        qprof.get_queries()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = qprof.get_queries()
        html_file = open("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_queries_1.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_queries_1.html

    Executing a QPROF step
    ^^^^^^^^^^^^^^^^^^^^^^^

    Numerous QPROF steps are accessible by directly
    using the corresponding methods. For instance,
    step 0 corresponds to the Vertica version, which
    can be executed using the associated method
    ``get_version``.

    .. ipython:: python
        :okwarning:

        qprof.get_version()

    .. note::

        To explore all available methods, please
        refer to the 'Methods' section. For
        additional information, you can also
        utilize the ``help`` function.

    It is possible to access the same
    step by using the ``step`` method.

    .. ipython:: python
        :okwarning:

        qprof.step(idx = 0)

    .. note::

        By changing the ``idx`` value above, you
        can check out all the steps of the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.

    **SQL Query**

    SQL query can be conveniently reproduced
    in a color formatted version:

    .. code-block:: python

        qprof.get_request()

    Query Performance Details
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    **Query Execution Time**

    To get the execution time of the entire query:

    .. ipython:: python
        :okwarning:

        qprof.get_qduration(unit="s")

    .. note::

        You can change the unit to
        "m" to get the result in
        minutes.

    **Query Execution Time Plots**

    To get the time breakdown of all the
    steps in a graphical output, we can call
    the ``get_qsteps`` attribute.

    .. code-block:: python

        qprof.get_qsteps(kind="bar")

    .. ipython:: python
        :suppress:
        :okwarning:

        import verticapy as vp
        vp.set_option("plotting_lib", "highcharts")
        fig = qprof.get_qsteps(kind="bar")
        html_text = fig.htmlcontent.replace("container", "performance_vertica_query_profiler_pie_plot")
        with open("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_pie_plot.html", "w") as file:
            file.write(html_text)

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_pie_plot.html

    .. note::

        The same plot can also be plotted using
        a bar plot by setting ``kind='bar'``.

    .. note::

        For charts, it is possible
        to pass many parameters to
        customize them. Example:
        You can use ``categoryorder``
        to sort the chart or ``width``
        and ``height`` to manage the
        size.

    **Query Plan**

    To get the entire query plan:

    .. ipython:: python

        qprof.get_qplan()

    **Query Plan Tree**

    We can easily call the function
    to get the query plan Graphviz:

    .. ipython:: python

        qprof.get_qplan_tree(return_graphviz = True)

    We can conveniently get the Query Plan tree:

    .. code-block::

        qprof.get_qplan_tree()

    .. ipython:: python
        :suppress:

        res = qprof.get_qplan_tree()
        res.render(filename='figures/performance_get_qplan_tree_1', format='png')


    .. image:: /../figures/performance_get_qplan_tree_1.png

    We can easily customize the tree:

    .. code-block::

        qprof.get_qplan_tree(
            metric='cost',
            shape='square',
            color_low='#0000FF',
            color_high='#FFC0CB',
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        res = qprof.get_qplan_tree(
            metric='cost',
            shape='square',
            color_low='#0000FF',
            color_high='#FFC0CB',
        )
        res.render(filename='figures/performance_get_qplan_tree_2', format='png')


    .. image:: /../figures/performance_get_qplan_tree_2.png

    We can look at a specific path ID,
    and look at some specific paths
    information:

    .. code-block::

        qprof.get_qplan_tree(
            path_id=1,
            path_id_info=[1, 3],
            metric='cost',
            shape='square',
            color_low='#0000FF',
            color_high='#FFC0CB',
        )

    .. ipython:: python
        :suppress:
        :okwarning:

        res = qprof.get_qplan_tree(
            path_id=1,
            path_id_info=[1, 3],
            metric='cost',
            shape='square',
            color_low='#0000FF',
            color_high='#FFC0CB',
        )
        res.render(filename='figures/performance_get_qplan_tree_3', format='png')


    .. image:: /../figures/performance_get_qplan_tree_3.png

    **Query Plan Profile**

    To visualize the time consumption
    of query profile plan:

    .. code-block:: python

        qprof.get_qplan_profile(kind = "pie")

    .. ipython:: python
        :suppress:

        vp.set_option("plotting_lib", "plotly")
        fig = qprof.get_qplan_profile(kind="pie")
        fig.write_html("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_qplan_profile.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_qplan_profile.html

    .. note::

        The same plot can also be plotted using
        a bar plot by switching the ``kind``
        to "bar".

    **Query Events**

    We can easily look
    at the query events:

    .. code-block:: python

        qprof.get_query_events()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = qprof.get_query_events()
        html_file = open("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_query_events.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_query_events.html

    **CPU Time by Node and Path ID**

    Another very important metric could be the CPU time
    spent by each node. This can be visualized by:

    .. code-block:: python

        qprof.get_cpu_time(kind="bar")

    .. ipython:: python
        :suppress:

        fig = qprof.get_qplan_profile(kind="pie")
        fig.write_html("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cup_node.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cup_node.html

    In order to get the results in a tabular form,
    just switch the ``show`` option to ``False``.

    .. code-block:: python

        qprof.get_cpu_time(show=False)

    .. ipython:: python
        :suppress:
        :okwarning:

        result = qprof.get_cpu_time(show=False)
        html_file = open("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cpu_time_table.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cpu_time_table.html

    Query Execution Report
    ^^^^^^^^^^^^^^^^^^^^^^^

    To obtain a comprehensive performance report,
    including specific details such as which node
    executed each operation and the corresponding
    timing information, utilize the following syntax:

    .. code-block:: python

        qprof.get_qexecution_report()

    .. ipython:: python
        :suppress:

        result = qprof.get_qexecution_report()
        html_file = open("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_full_report.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_full_report.html

    Node/Cluster Information
    ^^^^^^^^^^^^^^^^^^^^^^^^^

    **Nodes**

    To get node-wise performance
    information, ``get_qexecution``
    can be used:

    .. code-block:: python

        qprof.get_qexecution()

    .. ipython:: python
        :suppress:

        import verticapy as vp
        vp.set_option("plotting_lib", "plotly")
        fig = qprof.get_qexecution()
        fig.write_html("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_qexecution_1.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_qexecution_1.html

    .. note::

        To use one specific node:

        .. code-block:: python

            qprof.get_qexecution(
                node_name = "v_vdash_node0003",
                metric = "exec_time_ms",
                kind = "pie",
            )

        To use multiple nodes:

        .. code-block:: python

            qprof.get_qexecution(
                node_name = [
                    "v_vdash_node0001",
                    "v_vdash_node0003",
                ],
                metric = "exec_time_ms",
                kind = "pie",
            )

        The node name is different for different
        configurations. You can search for the node
        names in the full report.

    **Cluster**

    To get cluster configuration
    details, we can use:

    .. code-block:: python

        qprof.get_cluster_config()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = qprof.get_cluster_config()
        html_file = open("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cluster_table.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cluster_table.html

    The Cluster Report can also
    be conveniently extracted:

    .. code-block:: python

        qprof.get_rp_status()

    .. ipython:: python
        :suppress:
        :okwarning:

        result = qprof.get_rp_status()
        html_file = open("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cluster_table_2.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cluster_table_2.html

    .. important::

        Each method may have multiple
        parameters and options. It is
        essential to refer to the
        documentation of each method
        to understand its usage.
    """

    # Init Functions

    @save_verticapy_logs
    def __init__(
        self,
        transactions: Union[None, str, list[int], list[tuple[int, int]]] = None,
        key_id: Optional[str] = None,
        resource_pool: Optional[str] = None,
        target_schema: Union[None, str, dict] = None,
        overwrite: bool = False,
        add_profile: bool = True,
        check_tables: bool = True,
        iterchecks: bool = False,
    ) -> None:
        # TRANSACTIONS ARE STORED AS A LIST OF (tr_id, st_id) AND
        # AN INDEX USED TO NAVIGATE THROUGH THE DIFFERENT tuples.
        self.transactions = []
        self.transactions_idx = 0

        # List of queries to execute.
        requests = []

        # CASE WHEN TUPLE OF TWO ELEMENTS: (tr_id, st_id)
        if isinstance(transactions, tuple) and len(transactions) == 2:
            self.transactions = [transactions]

        # CASE WHEN LIST OF tr_id OR LIST OF (tr_id, st_id) OR queries
        # IT CAN ALSO BE A COMBINATION OF THE THREE TYPES.
        elif isinstance(transactions, list):
            for tr in transactions:
                if isinstance(tr, int):
                    self.transactions += [(tr, 1)]
                elif isinstance(tr, tuple):
                    if (
                        len(tr) == 2
                        and isinstance(tr[0], int)
                        and isinstance(tr[1], int)
                    ):
                        self.transactions += [tr]
                elif isinstance(tr, str):
                    requests += [tr]
                else:
                    raise TypeError(
                        f"Wrong type inside {transactions}. Expecting: "
                        f"int, tuple or str.\nFound {type(tr)}."
                    )

        # CASE WHEN integer
        elif isinstance(transactions, int):
            self.transactions = [(transactions, 1)]  # DEFAULT STATEMENT: 1

        # CASE WHEN str
        elif isinstance(transactions, str):
            requests += [transactions]

        elif isinstance(transactions, NoneType) and (
            isinstance(key_id, NoneType) or isinstance(target_schema, NoneType)
        ):
            raise ValueError(
                "When 'transactions' is not defined, a 'key_id' and a "
                "'target_schema' must be defined to retrieve all the "
                "transactions."
            )

        elif isinstance(transactions, NoneType):
            ...

        else:
            raise TypeError(
                "Wrong type for parameter 'transactions'. Expecting "
                "one of the following types: tuple[int, int] | "
                "list of tuple[int, int] | integer | list of integer "
                f"| string | list of strings.\nFound {type(transactions)}."
            )

        # CHECKING key_id; CREATING ONE IF IT DOES NOT EXIST.
        if isinstance(key_id, NoneType) or (
            not (isinstance(transactions, NoneType)) and not (overwrite)
        ):
            if not (isinstance(key_id, NoneType)) and (
                not (isinstance(transactions, NoneType)) and not (overwrite)
            ):
                warning_message = (
                    f"Parameter 'transactions' is not None, "
                    "'key_id' is defined and parameter 'overwrite' "
                    "is set to False. It means you are trying to "
                    "use a potential existing key to store new "
                    "transactions. This operation is not yet "
                    "supported. A new key will be then generated."
                )
                warnings.warn(warning_message, Warning)
            self.key_id = str(uuid.uuid1()).replace("-", "")
        else:
            if isinstance(key_id, int):
                self.key_id = str(key_id)
            elif isinstance(key_id, str):
                self.key_id = key_id
            else:
                raise TypeError(
                    "Wrong type for parameter 'key_id'. Expecting "
                    f"an integer or a string. Found {type(key_id)}."
                )

        # LOOKING AT A POSSIBLE QUERY TO EXECUTE.
        if len(requests) > 0:
            for request in requests:
                if not (isinstance(resource_pool, NoneType)):
                    _executeSQL(
                        f"SET SESSION RESOURCE POOL {resource_pool} ;",
                        title="Setting the resource pool.",
                        method="cursor",
                    )
                if add_profile:
                    fword = clean_query(request).strip().split()[0].lower()
                    if fword != "profile":
                        request = "PROFILE " + request
                _executeSQL(
                    request,
                    title="Executing the query.",
                    method="cursor",
                )
                query = """
                    SELECT
                        transaction_id,
                        statement_id
                    FROM QUERY_REQUESTS 
                    WHERE session_id = (SELECT current_session())
                      AND is_executing='f'
                    ORDER BY start_timestamp DESC LIMIT 1;"""
                res = _executeSQL(
                    query,
                    title="Getting transaction_id, statement_id.",
                    method="fetchrow",
                )
                if not isinstance(res, NoneType):
                    transaction_id, statement_id = res[0], res[1]
                    self.transactions += [(transaction_id, statement_id)]
                else:
                    warning_message = (
                        f"No transaction linked to query: {request}."
                        "\nIt might be still running. This transaction"
                        " will be skipped."
                    )
                    warnings.warn(warning_message, Warning)

        if len(self.transactions) == 0 and isinstance(key_id, NoneType):
            raise ValueError("No transactions found.")

        elif len(self.transactions) != 0:
            if (len(self.transactions[0]) > 0) and isinstance(
                self.transactions[0][0], int
            ):
                self.transaction_id = self.transactions[0][0]
            else:
                self.transaction_id = 1
            if (len(self.transactions[0]) > 1) and isinstance(
                self.transactions[0][1], int
            ):
                self.statement_id = self.transactions[0][1]
            else:
                self.statement_id = 1

        # BUILDING THE target_schema.
        if target_schema == "v_temp_schema":
            self.target_schema = self._v_temp_schema_dict()
        else:
            if isinstance(target_schema, str):
                self.target_schema = {}
                for schema in self._v_temp_schema_dict():
                    self.target_schema[schema] = target_schema
            else:
                self.target_schema = copy.deepcopy(target_schema)

        self.overwrite = overwrite
        self._create_copy_v_table()

        # SETTING THE requests AND queries durations.
        if conf.get_option("print_info"):
            print("Setting the requests and queries durations...")
        self._set_request_qd()

        # WARNING MESSAGES.
        if check_tables:
            self._check_v_table(iterchecks=iterchecks)

    # Tools

    def _check_kind(self, kind: str, kind_list: list) -> str:
        """
        Checks if the parameter 'kind'
        is correct and returns the
        corrected version of it.
        """
        kind = str(kind).lower()
        if kind not in kind_list:
            raise ValueError(
                "Parameter Error, 'kind' should be in "
                f"[{' | '.join(kind_list)}].\nFound {kind}."
            )
        return kind

    def _check_vdf_empty(self, vdf: vDataFrame) -> Literal[True]:
        """
        Checks if the vDataFrame is empty and
        raises the appropriate error message.
        """
        n, m = vdf.shape()
        if m == 0:
            raise EmptyParameter(
                "Failed to generate the final chart. Please check for any "
                "errors or issues with the data, and ensure that all required "
                "parameters are correctly set.\n"
                "Something abnormal happened. The vDataFrame seems to have "
                "no columns. This can occur if there was an error in the "
                "data ingestion or if the tables were modified after being "
                "ingested."
            )
        elif n == 0:
            raise EmptyParameter(
                "Failed to generate the final chart. Please check for any "
                "errors or issues with the data, and ensure that all required "
                "parameters are correctly set.\n"
                "The performance data needed to execute the operation is "
                "empty. This suggests that the information related to the "
                "specific transaction may not have been stored properly or "
                "might have been erased. We recommend saving this information "
                "in a different schema and multiple times to avoid any loss."
            )
        return True

    def _get_interval_str(self, unit: Literal["s", "m", "h"]) -> str:
        """
        Converts the input str to the
        corresponding interval.
        """
        unit = str(unit).lower()
        if unit.startswith("s"):
            div = "00:00:01"
        elif unit.startswith("m"):
            div = "00:01:00"
        elif unit.startswith("h"):
            div = "01:00:00"
        else:
            raise ValueError("Incorrect parameter 'unit'.")
        return div

    def _get_interval(self, unit: Literal["s", "m", "h"]) -> int:
        """
        Converts the input str to the
        corresponding integer.
        """
        unit = str(unit).lower()
        if unit.startswith("s"):
            div = 1000000
        elif unit.startswith("m"):
            div = 60000000
        elif unit.startswith("h"):
            div = 3600000000
        else:
            raise ValueError("Incorrect parameter 'unit'.")
        return div

    def _get_chart_method(
        self,
        v_object: Any,
        kind: Literal[
            "bar",
            "barh",
            "pie",
        ],
    ) -> Callable:
        """
        Returns the input object
        chart method: The one to
        use to draw the final
        graphic.
        """
        kind = str(kind).lower()
        if kind == "pie":
            return v_object.pie
        elif kind == "bar":
            return v_object.bar
        elif kind == "barh":
            return v_object.barh
        else:
            ValueError("Incorrect parameter 'kind'.")

    def _replace_schema_in_query(self, query: SQLExpression) -> SQLExpression:
        """
        Map all the relations in the
        query to the current ones.
        """
        if not (hasattr(self, "target_schema")) or isinstance(
            self.target_schema, NoneType
        ):
            return query
        fquery = copy.deepcopy(query)
        for sch in self.target_schema:
            fquery = fquery.replace(sch, self.target_schema[sch])
        for table in self.target_tables:
            fquery = fquery.replace(table, self.target_tables[table])
        return fquery

    @staticmethod
    def _v_temp_schema_dict() -> dict:
        """
        Tables used by the ``QueryProfiler``
        object to link the main relation to
        the temporary schema.
        """
        return {
            "v_internal": "v_temp_schema",
            "v_monitor": "v_temp_schema",
        }

    @staticmethod
    def _v_table_dict() -> dict:
        """
        Tables used by the ``QueryProfiler``
        object and their linked schema.
        """
        return {
            "dc_requests_issued": "v_internal",
            "dc_query_executions": "v_internal",
            "dc_explain_plans": "v_internal",
            "execution_engine_profiles": "v_monitor",
            "query_events": "v_monitor",
            "query_plan_profiles": "v_monitor",
            "query_profiles": "v_monitor",
            "resource_pool_status": "v_monitor",
            "host_resources": "v_monitor",
            # New Tables - still not used.
            "dc_plan_activities": "v_internal",
            "dc_lock_attempts": "v_internal",
            "dc_plan_resources": "v_internal",
            "configuration_parameters": "v_monitor",
            "projection_storage": "v_monitor",
            "projection_usage": "v_monitor",
            "query_consumption": "v_monitor",
            "query_events": "v_monitor",
            "resource_acquisitions": "v_monitor",
            "storage_containers": "v_monitor",
            "projections": "v_catalog",
            "projection_columns": "v_catalog",
            "resource_pools": "v_catalog",
        }

    @staticmethod
    def _v_config_table_list() -> list:
        """
        Config Tables do not use a
        ``transaction_id`` and a
        ``statement_is``.
        """
        return [
            "dc_lock_attempts",
            "configuration_parameters",
            "projections",
            "projection_columns",
            "projection_storage",
            "resource_pools",
            "resource_pool_status",
            "host_resources",
            "storage_containers",
        ]

    def _create_copy_v_table(self) -> None:
        """
        Functions to create a copy
        of the performance tables.
        If the tables exist, it
        will use them to do the
        profiling.
        """
        self.v_tables_dtypes = []
        self.tables_dtypes = []
        target_tables = {}
        v_temp_table_dict = self._v_table_dict()
        v_config_table_list = self._v_config_table_list()
        loop = v_temp_table_dict.items()
        if conf.get_option("print_info"):
            print("Searching the performance tables...")
        if conf.get_option("tqdm"):
            loop = tqdm(loop, total=len(loop))
        idx = 0
        for table, schema in loop:
            sql = "CREATE "
            exists = True
            if (
                not (isinstance(self.target_schema, NoneType))
                and schema in self.target_schema
            ):
                new_schema = self.target_schema[schema]
                new_table = f"qprof_{table}_{self.key_id}"
                if table == "dc_requests_issued" and len(self.transactions) == 0:
                    self.transactions = _executeSQL(
                        f"""SELECT 
                                transaction_id, 
                                statement_id 
                            FROM {new_schema}.{new_table}
                            ORDER BY 
                            time DESC,
                            transaction_id DESC,
                            statement_id DESC;""",
                        title="Getting the transactions and statement ids.",
                        method="fetchall",
                    )
                    self.transactions = [tuple(tr) for tr in self.transactions]
                    if len(self.transactions) == 0:
                        raise ValueError("No transactions found.")
                    self.transaction_id = self.transactions[0][0]
                    self.statement_id = self.transactions[0][1]
                    if isinstance(self.transaction_id, NoneType):
                        self.transaction_id = self.transactions[0][0]
                    if isinstance(self.statement_id, NoneType):
                        self.statement_id = self.transactions[0][1]
                if new_schema == "v_temp_schema":
                    sql += f"LOCAL TEMPORARY TABLE {new_table} ON COMMIT PRESERVE ROWS "
                else:
                    sql += f"TABLE {new_schema}.{new_table}"
                sql += f" AS SELECT * FROM {schema}.{table}"
                if table not in v_config_table_list:
                    sql += " WHERE "
                    jdx = 0
                    for tr, st in self.transactions:
                        if jdx > 0:
                            sql += " OR "
                        sql += f"(transaction_id={tr} AND statement_id={st})"
                        jdx += 1
                    sql += " ORDER BY transaction_id, statement_id"
                target_tables[table] = new_table

                # Getting the new DATATYPES
                try:
                    if not (self.overwrite):
                        self.tables_dtypes += [
                            get_data_types(
                                f"SELECT * FROM {new_schema}.{new_table} LIMIT 0",
                            )
                        ]
                except:
                    if conf.get_option("print_info") and idx == 0:
                        print("Some tables seem to not exist...")
                        print("Creating a copy of the performance tables...\n")
                        print(
                            f"The key used to build up the tables is: {self.key_id}\n"
                        )
                        print("You can access the key by using the 'key_id' attribute.")
                    exists = False
                    idx += 1

                # Getting the Performance tables DATATYPES
                self.v_tables_dtypes += [
                    get_data_types(
                        f"SELECT * FROM {schema}.{table} LIMIT 0",
                    )
                ]
                if not (exists) or (self.overwrite):
                    self.tables_dtypes += [self.v_tables_dtypes[-1]]

                if not (exists) or (self.overwrite):
                    if conf.get_option("print_info"):
                        print(
                            f"Copy of {schema}.{table} created in {new_schema}.{new_table}"
                        )
                    try:
                        if self.overwrite:
                            _executeSQL(
                                f"DROP TABLE IF EXISTS {new_schema}.{new_table}",
                                title="Dropping the performance table.",
                            )
                        _executeSQL(
                            sql,
                            title="Creating performance tables.",
                        )
                    except Exception as e:
                        warning_message = (
                            "An error occurs during the creation "
                            f"of the relation {new_schema}.{new_table}.\n"
                            "Tips: To overwrite the tables, set the parameter "
                            "overwrite=True.\nYou can also set create_table=False"
                            " to skip the table creation and to use the existing "
                            "ones.\n\nError Details:\n" + str(e)
                        )
                        warnings.warn(warning_message, Warning)
        self.target_tables = target_tables

    def _check_v_table(self, iterchecks: bool = True) -> None:
        """
        Checks if all the transactions
        exist in all the different
        tables.

        Parameters
        ----------
        iterchecks: bool, optional
            If set to ``True``, the checks are done
            iteratively instead of using a unique
            SQL query. Usually checks are faster
            when this parameter is set to ``False``.

            .. note::

                This parameter is used only when
                ``check_tables is True``.
        """
        tables = list(self._v_table_dict().keys())
        tables_schema = self._v_table_dict()
        config_table = self._v_config_table_list()
        warning_message = ""
        loop = self.transactions
        if iterchecks:
            if conf.get_option("tqdm"):
                loop = tqdm(loop, total=len(loop))
            if conf.get_option("print_info"):
                print("Checking all the tables consistency iteratively...")
            for tr_id, st_id in loop:
                for table_name in tables:
                    if table_name not in config_table:
                        if len(self.target_tables) == 0:
                            sc, tb = tables_schema[table_name], table_name
                        else:
                            tb = self.target_tables[table_name]
                            schema = tables_schema[table_name]
                            sc = self.target_schema[schema]
                        query = f"""
                            SELECT
                                transaction_id,
                                statement_id
                            FROM {sc}.{tb}
                            WHERE
                                transaction_id = {tr_id}
                            AND statement_id = {st_id}
                            LIMIT 1"""
                        res = _executeSQL(
                            query,
                            title=f"Checking transaction: ({tr_id}, {st_id}); relation: {sc}.{tb}.",
                            method="fetchall",
                        )
                        if not (res):
                            warning_message += f"({tr_id}, {st_id}) -> {sc}.{tb}\n"
        else:
            select = ["q0.transaction_id", "q0.statement_id"]
            jointables, relations = [], []
            for idx, table_name in enumerate(tables):
                if table_name not in config_table:
                    if len(self.target_tables) == 0:
                        sc, tb = tables_schema[table_name], table_name
                    else:
                        tb = self.target_tables[table_name]
                        schema = tables_schema[table_name]
                        sc = self.target_schema[schema]
                    select += [f"q{idx}.row_count AS {tb}"]
                    current_table = (
                        "(SELECT transaction_id, statement_id, COUNT(*) AS"
                        f" row_count FROM {sc}.{tb} GROUP BY 1,2) q{idx}"
                    )
                    if idx != 0:
                        current_table += " USING (transaction_id, statement_id)"
                    jointables += [current_table]
                    relations += [f"{sc}.{tb}"]
            if conf.get_option("print_info"):
                print("Checking all the tables consistency using a single SQL query...")
            query = (
                "SELECT "
                + ", ".join(select)
                + " FROM "
                + " FULL JOIN ".join(jointables)
            )
            res = _executeSQL(
                query,
                title=f"Checking all transactions.",
                method="fetchall",
            )
            if len(res) == 0:
                warning_message = (
                    "No transaction found. Please check the system tables."
                )
            else:
                n = len(res[0])
                transactions_dict = {}
                for row in res:
                    transactions_dict[(row[0], row[1])] = {}
                    for idx in range(2, n):
                        transactions_dict[(row[0], row[1])][relations[idx - 2]] = row[
                            idx
                        ]
                for tr_id, st_id in loop:
                    if (tr_id, st_id) not in transactions_dict:
                        warning_message += f"({tr_id}, {st_id}) -> all tables\n"
                    else:
                        for rel in relations:
                            nb_elem = transactions_dict[(tr_id, st_id)][rel]
                            if isinstance(nb_elem, NoneType) or nb_elem == 0:
                                warning_message += f"({tr_id}, {st_id}) -> {rel}\n"
        if len(warning_message) > 0:
            warning_message = "\nSome transactions are missing:\n\n" + warning_message
        missing_column = ""
        inconsistent_dt = ""
        table_name_list = list(self._v_table_dict())
        n = len(self.tables_dtypes)
        if conf.get_option("print_info"):
            print("Checking all the tables data types...")
        for i in range(n):
            table_name = table_name_list[i]
            table_1 = self.v_tables_dtypes[i]
            table_2 = self.tables_dtypes[i]
            for col_1, dt_1 in table_1:
                is_in = False
                for col_2, dt_2 in table_2:
                    if col_2.lower() == col_1.lower():
                        is_in = True
                        if dt_1 != dt_2:
                            inconsistent_dt += (
                                f"{table_name} | {col_1}: {dt_1} -> {dt_2}\n"
                            )
                        break
                if not (is_in):
                    missing_column += f"{table_name} | {col_1}\n"
        if missing_column:
            warning_message += "\nSome columns are missing:\n\n" + missing_column + "\n"
        if inconsistent_dt:
            warning_message += (
                "\nSome data types are inconsistent:\n\n" + inconsistent_dt + "\n"
            )
        if len(warning_message) > 0:
            warning_message += (
                "This could potentially lead to incorrect computations or "
                "errors. Please review the various tables and investigate "
                "why this data was modified or is missing. It may have "
                "been wrongly imported, accidentally deleted or automatically "
                "removed, especially if you are directly working on the "
                "performance tables."
            )
            warnings.warn(warning_message, Warning)

    def _set_request_qd(self):
        """
        Computes and sets the current
        ``transaction_id`` requests.
        """
        self.requests = []
        self.request_labels = []
        self.qdurations = []
        query = f"""
            SELECT 
                q0.transaction_id, 
                q0.statement_id, 
                request, 
                label, 
                query_duration_us
            FROM 
            v_internal.dc_requests_issued AS q0
            FULL JOIN
            v_monitor.query_profiles AS q1 
            USING (transaction_id, statement_id);"""
        query = self._replace_schema_in_query(query)
        res = _executeSQL(
            query,
            title="Getting the corresponding query",
            method="fetchall",
        )
        transactions_dict = {}
        for row in res:
            transactions_dict[(row[0], row[1])] = {
                "request": row[2],
                "label": row[3],
                "query_duration_us": row[4],
            }
        for tr_id, st_id in self.transactions:
            if (tr_id, st_id) not in transactions_dict:
                raise QueryError(
                    f"No transaction with transaction_id={tr_id} "
                    f"and statement_id={st_id} was found in the "
                    "v_internal.dc_requests_issued table."
                )
            else:
                info = transactions_dict[(tr_id, st_id)]
                self.requests += [info["request"]]
                self.request_labels += [info["label"]]
                self.qdurations += [info["query_duration_us"]]
        self.request = self.requests[self.transactions_idx]
        self.qduration = self.qdurations[self.transactions_idx]

    # Navigation

    def set_position(self, idx: Union[int, tuple]) -> None:
        """
        A utility function to utilize
        a specific transaction from
        the ``QueryProfiler`` stack.
        """
        n = len(self.transactions)
        if isinstance(idx, int) and not (0 <= idx < n):
            raise ValueError(f"Incorrect index, it should be between 0 and {n - 1}")
        else:
            if isinstance(idx, tuple):
                if len(idx) != 2 or (
                    not (isinstance(idx[0], int)) or not (isinstance(idx[1], int))
                ):
                    raise ValueError(
                        "When 'idx' is a tuple, it should be made of two integers."
                    )
                found = False
                for j, tr in enumerate(self.transactions):
                    if tr == idx:
                        idx = j
                        found = True
                        break
                if not (found):
                    raise ValueError(f"Transaction not found: {idx}.")
            if isinstance(idx, int):
                self.transactions_idx = idx
                self.transaction_id = self.transactions[idx][0]
                self.statement_id = self.transactions[idx][1]
                self.request = self.requests[idx]
                self.qduration = self.qdurations[idx]
            else:
                raise TypeError(
                    "Wrong type for parameter 'idx'. Expecting: int or tuple."
                    f"\nFound: {type(idx)}."
                )

    def next(self) -> None:
        """
        A utility function to utilize
        the next transaction from
        the ``QueryProfiler`` stack.
        """
        idx = self.transactions_idx
        n = len(self.transactions)
        if idx + 1 == n:
            idx = 0
        else:
            idx = idx + 1
        self.transactions_idx = idx
        self.transaction_id = self.transactions[idx][0]
        self.statement_id = self.transactions[idx][1]
        self.request = self.requests[idx]
        self.qduration = self.qdurations[idx]

    def previous(self) -> None:
        """
        A utility function to utilize
        the previous transaction from
        the ``QueryProfiler`` stack.
        """
        idx = self.transactions_idx
        n = len(self.transactions)
        if idx - 1 == -1:
            idx = n - 1
        else:
            idx = idx - 1
        self.transactions_idx = idx
        self.transaction_id = self.transactions[idx][0]
        self.statement_id = self.transactions[idx][1]
        self.request = self.requests[idx]
        self.qduration = self.qdurations[idx]

    # Main Method

    def step(self, idx: int, *args, **kwargs) -> Any:
        """
        Function to return the
        QueryProfiler Step.
        """
        steps_id = {
            0: self.get_version,
            1: self.get_request,
            2: self.get_qduration,
            3: self.get_qsteps,
            4: NotImplemented,
            5: self.get_qplan_tree,
            6: self.get_qplan_profile,
            7: NotImplemented,
            8: NotImplemented,
            9: NotImplemented,
            10: self.get_query_events,
            11: NotImplemented,
            12: self.get_cpu_time,
            13: NotImplemented,
            14: self.get_qexecution,
            15: NotImplemented,
            16: NotImplemented,
            17: NotImplemented,
            18: NotImplemented,
            19: NotImplemented,
            20: self.get_rp_status,
            21: self.get_cluster_config,
        }
        return steps_id[idx](*args, **kwargs)

    # Perf/Query Tables

    def get_queries(self) -> vDataFrame:
        """
        Returns all the queries
        and their respective information,
        of a ``QueryProfiler`` object.

        Returns
        -------
        vDataFrame
            queries information.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can easily look at all
        the transactions:

        .. code-block:: python

            qprof.get_queries()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_queries_1.html

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        n = len(self.transactions)
        current_query = [
            (self.transaction_id, self.statement_id) == tr for tr in self.transactions
        ]

        return vDataFrame(
            {
                "index": [i for i in range(n)],
                "is_current": current_query,
                "transaction_id": [tr[0] for tr in self.transactions],
                "statement_id": [tr[1] for tr in self.transactions],
                "request_label": copy.deepcopy(self.request_labels),
                "request": copy.deepcopy(self.requests),
                "qduration": [qd / 1000000 for qd in self.qdurations],
            },
            _clean_query=False,
        )

    def get_table(self, table_name: Optional[str] = None) -> Union[list, vDataFrame]:
        """
        Returns the associated Vertica
        Table. This function allows easy
        data exploration of all the
        performance meta-tables.

        Parameters
        ----------
        table_name: str, optional
            Vertica DC Table to return.
            If empty, the list of all
            the tables is returned to
            simplify the selection.

            Examples:
             - execution_engine_profiles
             - dc_explain_plans
             - dc_query_executions
             - dc_requests_issued
             - host_resources
             - query_plan_profiles
             - query_profiles
             - resource_pool_status

        Returns
        -------
        vDataFrame
            Vertica DC Table.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can easily look at any Vertica
        Performance Tables easily:

        .. code-block:: python

            qprof.get_table('dc_requests_issued')

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_table_1.html

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        tables = list(self._v_table_dict().keys())
        if isinstance(table_name, NoneType):
            return tables
        elif table_name not in tables:
            raise ValueError(
                f"Did not find the Query Profiler Table {table_name}.\n"
                f"Please choose one in {', '.join(tables)}"
            )
        else:
            tables_schema = self._v_table_dict()
            if len(self.target_tables) == 0:
                sc, tb = tables_schema[table_name], table_name
            else:
                tb = self.target_tables[table_name]
                schema = tables_schema[table_name]
                sc = self.target_schema[schema]
            return vDataFrame(f"SELECT * FROM {sc}.{tb}")

    # Steps

    # Step 0: Vertica Version
    @staticmethod
    def get_version() -> tuple[int, int, int, int]:
        """
        Returns the current Vertica version.

        Returns
        -------
        tuple
            List containing the version information.
            ``MAJOR, MINOR, PATCH, POST``

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. ipython:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. ipython:: python
            :okwarning:

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can get the Verica version by:

        .. ipython:: python

            qprof.get_version()

        .. note::

            When using this function in a Jupyter environment
            you should be able to see the SQL query nicely formatted
            as well as color coded for ease of readability.

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        return vertica_version()

    # Step 1: Query text
    def get_request(
        self,
        indent_sql: bool = True,
        print_sql: bool = True,
        return_html: bool = False,
    ) -> str:
        """
        Returns the query linked to the object with the
        specified transaction ID and statement ID.

        Parameters
        ----------
        indent_sql: bool, optional
            If set to true, indents the SQL code.
        print_sql: bool, optional
            If set to true, prints the SQL code.
        return_html: bool, optional
            If set to true, returns the query HTML
            code.

        Returns
        -------
        str
            SQL query.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can get the SQL query by:

        .. ipython:: python
            :okwarning:

            qprof.get_request()

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        res = format_query(
            query=self.request, indent_sql=indent_sql, print_sql=print_sql
        )
        if return_html:
            return format_query(
                query=self.request,
                indent_sql=indent_sql,
                print_sql=print_sql,
                only_html=True,
            )
        return res[0]

    # Step 2: Query duration
    def get_qduration(
        self,
        unit: Literal["s", "m", "h"] = "s",
    ) -> float:
        """
        Returns the Query duration.

        Parameters
        ----------
        unit: str, optional
            Time Unit.

            - s:
                second

            - m:
                minute

            - h:
                hour

        Returns
        -------
        float
            Query duration.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can get the execution time by:

        .. ipython:: python

            qprof.get_qduration(unit = "s")

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        return float(self.qduration / self._get_interval(unit))

    # Step 3: Query execution steps
    def get_qsteps(
        self,
        unit: Literal["s", "m", "h"] = "s",
        kind: Literal[
            "bar",
            "barh",
        ] = "bar",
        categoryorder: Literal[
            "trace",
            "category ascending",
            "category descending",
            "total ascending",
            "total descending",
            "min ascending",
            "min descending",
            "max ascending",
            "max descending",
            "sum ascending",
            "sum descending",
            "mean ascending",
            "mean descending",
            "median ascending",
            "median descending",
        ] = "sum descending",
        show: bool = True,
        **style_kwargs,
    ) -> Union[PlottingObject, vDataFrame]:
        """
        Returns the Query Execution Steps chart.

        Parameters
        ----------
        unit: str, optional
            Unit used to draw the chart.

            - s:
                second

            - m:
                minute

            - h:
                hour

        kind: str, optional
            Chart Type.

            - bar:
                Bar Chart.

            - barh:
                Horizontal Bar Chart.

        categoryorder: str, optional
            How to sort the bars.
            One of the following options:

            - trace (no transformation)
            - category ascending
            - category descending
            - total ascending
            - total descending
            - min ascending
            - min descending
            - max ascending
            - max descending
            - sum ascending
            - sum descending
            - mean ascending
            - mean descending
            - median ascending
            - median descending

        show: bool, optional
            If set to True, the Plotting object
            is returned.
        **style_kwargs
            Any  optional parameter to
            pass to the plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can get the time breakdown of all the
        steps in a graphical output, we can call
        the ``get_qsteps`` attribute.

        .. code-block:: python

            qprof.get_qsteps(kind="bar")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_pie_plot.html

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        if show:
            kind = self._check_kind(kind, ["bar", "barh"])
        div = self._get_interval_str(unit)
        query = f"""
            SELECT
                REPLACE(COALESCE(REGEXP_SUBSTR(
                    execution_step, '(.+):'), execution_step), ':', '') AS step,
                REPLACE(COALESCE(REGEXP_SUBSTR(
                    execution_step, ':(.+)'), execution_step), ':', '') AS substep,
                (completion_time - time) / '{div}'::interval AS elapsed
            FROM 
                v_internal.dc_query_executions 
            WHERE 
                transaction_id={self.transaction_id} AND 
                statement_id={self.statement_id}
            ORDER BY 2 DESC;"""
        query = self._replace_schema_in_query(query)
        vdf = vDataFrame(query)
        if show:
            self._check_vdf_empty(vdf)
            fun = self._get_chart_method(vdf, kind)
            return fun(
                columns=["step", "substep"],
                method="max",
                of="elapsed",
                categoryorder=categoryorder,
                max_cardinality=1000,
                kind="drilldown",
                **style_kwargs,
            )
        return vdf

    # Step 5: Query plan + EXPLAIN

    def get_qplan_explain(self, display_trees: bool = True) -> list:
        """
        TODO
        """
        return None
        # return parse_explain_graphviz(rows, display_trees=display_trees)

    def get_qplan_tr_order(
        self,
    ) -> list[int]:
        """
        Returns the Query Plan Temp
        Relations Order.
        It is used to sort correctly
        the temporary relations in
        the final Tree.

        Returns
        -------
        list
            List of the different
            temp relations index.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can easily call the function to get
        the final order:

            .. ipython:: python

                qprof._get_qplan_tr_order()

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        query = f"""
            SELECT 
                REGEXP_SUBSTR(step_label, '\\d+')::INT
            FROM v_internal.dc_plan_steps
            WHERE 
                transaction_id={self.transaction_id}
            AND statement_id={self.statement_id}
            AND step_label ILIKE '%TempRelWrite%' 
            ORDER BY step_key ASC;"""
        query = self._replace_schema_in_query(query)
        try:
            # TEST does not yet support this table.
            res = _executeSQL(
                query,
                title="Getting the corresponding query",
                method="fetchall",
            )
            return list(dict.fromkeys([q[0] for q in res]))
        except:
            return []

    def get_qplan(
        self,
        return_report: bool = False,
        print_plan: bool = True,
    ) -> Union[str, vDataFrame]:
        """
        Returns the Query Plan chart.

        Parameters
        ----------
        return_report: bool, optional
            If set to ``True``, the query
            plan report is returned.
        print_plan: bool, optional
            If set to ``True``, the query
            plan is printed.

        Returns
        -------
        str
            Query Plan.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can easily call the function to get the entire
        query plan:

            .. ipython:: python

                qprof.get_qplan()

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        query = f"""
            SELECT
                statement_id AS stmtid,
                path_id,
                path_line_index,
                path_line
            FROM 
                v_internal.dc_explain_plans
            WHERE 
                transaction_id={self.transaction_id}
            AND statement_id={self.statement_id}
            ORDER BY 
                statement_id,
                path_id,
                path_line_index;"""
        query = self._replace_schema_in_query(query)
        if not (return_report):
            path = _executeSQL(
                query,
                title="Getting the corresponding query",
                method="fetchall",
            )
            res = "\n".join([l[3] for l in path])
            if print_plan:
                print(res)
            return res
        vdf = vDataFrame(query).sort(["stmtid", "path_id", "path_line_index"])
        return vdf

    def get_qplan_tree(
        self,
        path_id: Optional[int] = None,
        path_id_info: Optional[list] = None,
        show_ancestors: bool = True,
        metric: Union[
            NoneType,
            str,
            tuple[str, str],
            list[str],
        ] = ["exec_time_ms", "prod_rows"],
        pic_path: Optional[str] = None,
        return_graphviz: bool = False,
        **tree_style,
    ) -> Union["Source", str]:
        """
        Draws the Query Plan tree.

        Parameters
        ----------
        path_id: int, optional
            A path ID used to filter
            the tree elements by
            starting from it.
        path_id_info: list, optional
            ``list`` of path_id used
            to display the different
            query information.
        show_ancestors: bool, optional
            If set to ``True`` the
            ancestors of ``path_id``
            are also displayed.
        metric: str | tuple | list, optional
            The metric used to color
            the tree nodes. One of
            the following:

            - None (no specific color)

            - bytes_spilled
            - clock_time_us
            - cost
            - cstall_us
            - exec_time_ms (default)
            - est_rows
            - mem_all_mb
            - mem_res_mb
            - proc_rows
            - prod_rows
            - pstall_us
            - rle_prod_rows
            - rows

            It can also be a ``list`` or
            a ``tuple`` of two metrics.

        pic_path: str, optional
            Absolute path to save
            the image of the tree.
        return_graphviz: bool, optional
            If set to ``True``, the
            ``str`` Graphviz tree is
            returned.
        tree_style: dict, optional
            ``dictionary`` used to
            customize the tree.

            - two_legend:
                If set to ``True``
                and two metrics are
                used, two legends will
                be drawn.
                Default: True
            - display_legend:
                If set to ``True``
                the legend is
                displayed.
                Default: True
            - color_low:
                Color used as the lower
                bound of the gradient.
                Default: '#00FF00' (green)
            - color_high:
                Color used as the upper
                bound of the gradient.
                Default: '#FF0000' (red)
            - fontcolor:
                Font color.
                Default (light-m): #000000 (black)
                Default (dark-m): #FFFFFF (white)
            - fontsize:
                Font size.
                Default: 22
            - fillcolor:
                Color used to fill the
                nodes in case no gradient
                is computed: ``metric=None``.
                Default (light-m): #FFFFFF (white)
                Default (dark-m): #000000 (black)
            - edge_color:
                Edge color.
                Default (light-m): #000000 (black)
                Default (dark-m): #FFFFFF (white)
            - edge_style:
                Edge Style.
                Default: 'solid'.
            - shape:
                Node shape.
                Default: 'circle'.
            - width:
                Node width.
                Default: 0.6.
            - height:
                Node height.
                Default: 0.6.
            - info_color:
                Color of the information box.
                Default: #DFDFDF (lightgray)
            - info_fontcolor:
                Fontcolor of the information
                box.
                Default: #000000 (black)
            - info_rowsize:
                Maximum size of a line
                in the information box.
                Default: 30
            - info_fontsize:
                Information box font
                size.
                Default: 8
            - storage_access:
                Maximum number of chars of
                the storage access box.
                Default: 9
            - network_edge:
                If set to ``True`` the
                network edges will all
                have their own style:
                dotted for BROADCAST,
                dashed for RESEGMENT
                else solid.
            - display_operator:
                If set to ``True`` the
                PATH ID operator of each
                node will be displayed.
            - display_operator_edge:
                If set to ``True`` the
                operator edge of each
                node will be displayed.
            - display_proj:
                If set to ``True`` the
                projection of each STORAGE
                ACCESS PATH ID will be
                partially displayed.
            - display_etc:
                If set to ``True`` and
                ``path_is is not None``
                the symbol "..." is used
                to represent the ancestors
                children when they have more
                than 1.
            - temp_relation_access:
                ``list`` of the temporary
                tables to display. ``main``
                represents the main relation
                plan.
                Ex: ``['TREL8', 'main']``
                will only display the
                temporary relation 8
                and the main relation.
                Default: []

        Returns
        -------
        graphviz.Source
            graphviz object.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can easily call the function
        to get the query plan Graphviz:

        .. ipython:: python

            qprof.get_qplan_tree(return_graphviz = True)

        We can conveniently get the Query Plan tree:

        .. code-block::

            qprof.get_qplan_tree()

        .. ipython:: python
            :suppress:

            res = qprof.get_qplan_tree()
            res.render(filename='figures/performance_get_qplan_tree_1', format='png')


        .. image:: /../figures/performance_get_qplan_tree_1.png

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        rows = self.get_qplan(print_plan=False)
        if len(rows) == "":
            raise ValueError("The Query Plan is empty. Its data might have been lost.")
        metric_value = {}
        if isinstance(metric, (str, NoneType)):
            metric = [metric]
        for me in metric:
            if me not in [None, "rows", "cost"]:
                vdf = self.get_qexecution_report()
                query = f"""
                    SELECT
                        path_id,
                        SUM({me})
                    FROM {vdf}
                    GROUP BY 1
                    ORDER BY 1"""
                res = _executeSQL(
                    query,
                    title="Getting the corresponding query",
                    method="fetchall",
                )
                metric_value[me] = {}
                for path_id_val, metric_val in res:
                    if not isinstance(metric_val, NoneType):
                        if me in [
                            "proc_rows",
                            "prod_rows",
                            "rows",
                            "rle_prod_rows",
                            "est_rows",
                        ]:  # Rows will always be integers
                            metric_value[me][path_id_val] = int(metric_val)
                        else:
                            metric_value[me][path_id_val] = float(metric_val)
                    else:
                        metric_value[me][path_id_val] = 0
        tree_style["temp_relation_order"] = self.get_qplan_tr_order()
        obj = PerformanceTree(
            rows,
            show_ancestors=show_ancestors,
            path_id_info=path_id_info,
            path_id=path_id,
            metric=metric,
            metric_value=metric_value,
            style=tree_style,
        )
        if return_graphviz:
            return obj.to_graphviz()
        return obj.plot_tree(pic_path)

    # Step 6: Query plan profile
    def get_qplan_profile(
        self,
        unit: Literal["s", "m", "h"] = "s",
        kind: Literal[
            "bar",
            "barh",
            "pie",
        ] = "pie",
        categoryorder: Literal[
            "trace",
            "category ascending",
            "category descending",
            "total ascending",
            "total descending",
        ] = "total descending",
        show: bool = True,
        **style_kwargs,
    ) -> Union[PlottingObject, vDataFrame]:
        """
        Returns the Query Plan chart.

        Parameters
        ----------
        unit: str, optional
            Unit used to draw the chart.

            - s:
                second.
            - m:
                minute.
            - h:
                hour.

        kind: str, optional
            Chart Type.

            - bar:
                Bar Chart.
            - barh:
                Horizontal Bar Chart.
            - pie:
                Pie Chart.

        categoryorder: str, optional
            How to sort the bars.
            One of the following options:

            - trace (no transformation)
            - category ascending
            - category descending
            - total ascending
            - total descending

        show: bool, optional
            If set to True, the Plotting object
            is returned.
        **style_kwargs
            Any  optional parameter to
            pass to the plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can visualize the query plan profile:

        .. code-block:: python

            qprof.get_qplan_profile(kind="pie")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_qplan_profile.html

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        if show:
            kind = self._check_kind(kind, ["bar", "barh", "pie"])
        div = self._get_interval_str(unit)
        where = ""
        if show:
            where = "AND running_time IS NOT NULL"
        query = f"""
            SELECT
                statement_id AS stmtid,
                path_id,
                path_line_index,
                running_time / '{div}'::interval AS running_time,
                (memory_allocated_bytes // (1024 * 1024))::numeric(18, 2) AS mem_mb,
                (read_from_disk_bytes // (1024 * 1024))::numeric(18, 2) AS read_mb,
                (received_bytes // (1024 * 1024))::numeric(18, 2) AS in_mb,
                (sent_bytes // (1024 * 1024))::numeric(18, 2) AS out_mb,
                left(path_line, 80) AS path_line
            FROM v_monitor.query_plan_profiles
            WHERE transaction_id={self.transaction_id} AND
                  statement_id={self.statement_id}{where}
            ORDER BY
                statement_id,
                path_id,
                path_line_index;"""
        query = self._replace_schema_in_query(query)
        vdf = vDataFrame(query).sort(["stmtid", "path_id", "path_line_index"])
        if show:
            self._check_vdf_empty(vdf)
            fun = self._get_chart_method(vdf["path_line"], kind)
            return fun(
                method="sum",
                of="running_time",
                categoryorder=categoryorder,
                max_cardinality=1000,
                **style_kwargs,
            )
        return vdf

    # Step 10: Query events
    def get_query_events(self) -> vDataFrame:
        """
        Returns a :py:class`vDataFrame`
        that contains a table listing
        query events.

        Returns
        -------
        A :py:class`vDataFrame` that
        contains a table listing query
        events.

        Columns:
          1) event_timestamp:
            - Type:
                Timestamp.
            - Description:
                When the event happened.
            - Example:
                ``2023-12-11 19:01:03.543272-05:00``
          2) node_name:
            - Type:
                string.
            - Description:
                Which node the event
                happened on.
            - Example:
                v_db_node0003
          3) event_category:
            - Type:
                string.
            - Description:
                The general kind of event.
            - Examples:
                OPTIMIZATION, EXECUTION.
          4) event_type:
            - Type:
                string.
            - Description:
                The specific kind of event.
            - Examples:
                AUTO_PROJECTION_USED, SMALL_MERGE_REPLACED.
          5) event_description:
            - Type:
                string.
            - Description:
                A sentence explaining the event.
            - Example:
                "The optimizer ran a query
                using auto-projections"
          6) operator_name:
            - Type:
                string.
            - Description:
                The name of the EE operator
                associated with this event.
                ``None`` if no operator is
                associated with the event.
            - Example:
                StorageMerge.
          7) path_id:
            - Type:
                integer.
            - Description:
                A number that uniquely
                identifies the operator
                in the plan.
            - Examples:
                1, 4...
          8) event_details:
            - Type:
                string.
            - Description:
                Additional specific
                information related
                to this event.
            - Example:
                "t1_super is an
                auto-projection"
          9) suggested_action:
            - Type:
                string.
            - Description:
                A sentence describing
                potential remedies.

        Examples
        --------
        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can look at the query events:

        .. code-block:: python

            qprof.get_query_events()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_query_events.html

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.QueryProfiler`.
        """
        query = f"""
            SELECT
                event_timestamp,
                node_name,
                event_category,
                event_type,
                event_description,
                operator_name,
                path_id,
                event_details,
                suggested_action
            FROM
                v_monitor.query_events
            WHERE
                transaction_id={self.transaction_id} AND
                statement_id={self.statement_id}
            ORDER BY
                1;"""
        query = self._replace_schema_in_query(query)
        vdf = vDataFrame(query)
        return vdf

    # Step 12: CPU Time by node and path_id
    def get_cpu_time(
        self,
        kind: Literal[
            "bar",
            "barh",
        ] = "bar",
        reverse: bool = False,
        categoryorder: Literal[
            "trace",
            "category ascending",
            "category descending",
            "total ascending",
            "total descending",
            "min ascending",
            "min descending",
            "max ascending",
            "max descending",
            "sum ascending",
            "sum descending",
            "mean ascending",
            "mean descending",
            "median ascending",
            "median descending",
        ] = "max descending",
        show: bool = True,
        **style_kwargs,
    ) -> Union[PlottingObject, vDataFrame]:
        """
        Returns the CPU Time by node and path_id chart.

        Parameters
        ----------
        kind: str, optional
            Chart Type.

            - bar:
                Bar Chart.
            - barh:
                Horizontal Bar Chart.

        reverse: bool, optional
            If set to ``True``, the
            chart will be reversed.
        categoryorder: str, optional
            How to sort the bars.
            One of the following options:

            - trace (no transformation)
            - category ascending
            - category descending
            - total ascending
            - total descending
            - min ascending
            - min descending
            - max ascending
            - max descending
            - sum ascending
            - sum descending
            - mean ascending
            - mean descending
            - median ascending
            - median descending

        show: bool, optional
            If set to ``True``, the
            Plotting object is returned.
        **style_kwargs
            Any  optional parameter to
            pass to the plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        --------

        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        To visualize the CPU time spent by each node:

        .. code-block:: python

            qprof.get_cpu_time(kind="bar")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cup_node.html

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        if show:
            kind = self._check_kind(kind, ["bar", "barh"])
        query = f"""
            SELECT 
                node_name, 
                path_id::VARCHAR, 
                counter_value
            FROM 
                v_monitor.execution_engine_profiles 
            WHERE 
                TRIM(counter_name) = 'execution time (us)' AND 
                transaction_id={self.transaction_id} AND 
                statement_id={self.statement_id}"""
        query = self._replace_schema_in_query(query)
        vdf = vDataFrame(query)
        columns = ["path_id", "node_name"]
        if reverse:
            columns.reverse()
        if show:
            self._check_vdf_empty(vdf)
            fun = self._get_chart_method(vdf, kind)
            return fun(
                columns=columns,
                method="SUM(counter_value) AS cet",
                categoryorder=categoryorder,
                max_cardinality=1000,
                **style_kwargs,
            )
        return vdf

    # Step 14A: Query execution report
    def get_qexecution_report(self) -> vDataFrame:
        """
        Returns the Query execution report.

        Returns
        -------
        vDataFrame
            report.

        Examples
        --------

        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        To get the complete execution report use:

        .. code-block:: python

            qprof.get_qexecution_report()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_full_report.html

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        query = f"""
            SELECT
                node_name,
                operator_name,
                path_id,
                ROUND(SUM(CASE TRIM(counter_name) WHEN 'execution time (us)' THEN
                    counter_value ELSE NULL END) / 1000, 3.0) AS exec_time_ms,
                SUM(CASE TRIM(counter_name) WHEN 'estimated rows produced' THEN
                    counter_value ELSE NULL END) AS est_rows,
                SUM(CASE TRIM(counter_name) WHEN 'rows processed' THEN
                    counter_value ELSE NULL END) AS proc_rows,
                SUM(CASE TRIM(counter_name) WHEN 'rows produced' THEN
                    counter_value ELSE NULL END) AS prod_rows,
                SUM(CASE TRIM(counter_name) WHEN 'rle rows produced' THEN
                    counter_value ELSE NULL END) AS rle_prod_rows,
                SUM(CASE TRIM(counter_name) WHEN 'consumer stall (us)' THEN
                    counter_value ELSE NULL END) AS cstall_us,
                SUM(CASE TRIM(counter_name) WHEN 'producer stall (us)' THEN
                    counter_value ELSE NULL END) AS pstall_us,
                SUM(CASE TRIM(counter_name) WHEN 'clock time (us)' THEN
                    counter_value ELSE NULL END) AS clock_time_us,
                ROUND(SUM(CASE TRIM(counter_name) WHEN 'memory reserved (bytes)' THEN
                    counter_value ELSE NULL END) / 1000000, 1.0) AS mem_res_mb,
                ROUND(SUM(CASE TRIM(counter_name) WHEN 'memory allocated (bytes)' THEN 
                    counter_value ELSE NULL END) / 1000000, 1.0) AS mem_all_mb,
                SUM(CASE TRIM(counter_name) WHEN 'bytes spilled' THEN
                    counter_value ELSE NULL END) AS bytes_spilled
            FROM
                v_monitor.execution_engine_profiles
            WHERE
                transaction_id={self.transaction_id} AND
                statement_id={self.statement_id} AND
                counter_value / 1000000 > 0
            GROUP BY
                1, 2, 3
            ORDER BY
                CASE WHEN SUM(CASE TRIM(counter_name) WHEN 'execution time (us)' THEN
                    counter_value ELSE NULL END) IS NULL THEN 1 ELSE 0 END ASC,
                5 DESC;"""
        query = self._replace_schema_in_query(query)
        return vDataFrame(query)

    # Step 14B: Query execution chart
    def get_qexecution(
        self,
        node_name: Union[None, str, list] = None,
        metric: str = "exec_time_ms",
        path_id: Optional[int] = None,
        kind: Literal[
            "bar",
            "barh",
            "pie",
        ] = "pie",
        multi: bool = True,
        categoryorder: Literal[
            "trace",
            "category ascending",
            "category descending",
            "total ascending",
            "total descending",
            "min ascending",
            "min descending",
            "max ascending",
            "max descending",
            "sum ascending",
            "sum descending",
            "mean ascending",
            "mean descending",
            "median ascending",
            "median descending",
        ] = "max descending",
        rows: int = 3,
        cols: int = 3,
        show: bool = True,
        **style_kwargs,
    ) -> Union[PlottingObject, vDataFrame]:
        """
        Returns the Query execution chart.

        Parameters
        ----------
        node_name: str | list, optional
            Node(s) name(s).
            It can be a simple node ``str``
            or a ``list`` of nodes.
            If empty, all the nodes are
            used.
        metric: str, optional
            Metric to use. One of the following:
            - all (all metrics are used).
            - bytes_spilled
            - clock_time_us
            - cstall_us
            - exec_time_ms (default)
            - est_rows
            - mem_all_mb
            - mem_res_mb
            - proc_rows
            - prod_rows
            - pstall_us
            - rle_prod_rows
        path_id: str
            Path ID.
        kind: str, optional
            Chart Type.

            - bar:
                Drilldown Bar Chart.
            - barh:
                Horizontal Drilldown
                Bar Chart.
            - pie:
                Pie Chart.

        multi: bool, optional
            If set to ``True``, a multi
            variable chart is drawn by
            using 'operator_name' and
            'path_id'. Otherwise, a
            single plot using 'operator_name'
            is drawn.
        categoryorder: str, optional
            How to sort the bars.
            One of the following options:

            - trace (no transformation)
            - category ascending
            - category descending
            - total ascending
            - total descending
            - min ascending
            - min descending
            - max ascending
            - max descending
            - sum ascending
            - sum descending
            - mean ascending
            - mean descending
            - median ascending
            - median descending

        rows: int, optional
            Only used when ``metric='all'``.
            Number of rows of the subplot.
        cols: int, optional
            Only used when ``metric='all'``.
            Number of columns of the subplot.
        show: bool, optional
            If set to ``True``, the
            Plotting object is returned.
        **style_kwargs
            Any  optional parameter to
            pass to the plotting functions.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        --------

        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        To get node-wise performance
        information, ``get_qexecution``
        can be used:

        .. code-block:: python

            qprof.get_qexecution()

        .. ipython:: python
            :suppress:

            import verticapy as vp
            vp.set_option("plotting_lib", "plotly")
            fig = qprof.get_qexecution()
            fig.write_html("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_qexecution_1.html")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_qexecution_1.html

        .. note::

            To use one specific node:

            .. code-block:: python

                qprof.get_qexecution(
                    node_name = "v_vdash_node0003",
                    metric = "exec_time_ms",
                    kind = "pie",
                )

            To use multiple nodes:

            .. code-block:: python

                qprof.get_qexecution(
                    node_name = [
                        "v_vdash_node0001",
                        "v_vdash_node0003",
                    ],
                    metric = "exec_time_ms",
                    kind = "pie",
                )

            The node name is different for different
            configurations. You can search for the node
            names in the full report.

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        if show:
            kind = self._check_kind(kind, ["bar", "barh", "pie"])
        if metric == "all" and show:
            if conf.get_option("plotting_lib") != "plotly":
                raise ExtensionError(
                    "Plots with metric='all' is only available for Plotly Integration."
                )
            figs = []
            all_metrics = QprofUtility._get_metrics()
            for metric in [None, "cost", "rows"]:
                all_metrics.remove(metric)
            for metric in all_metrics:
                figs += [
                    self.get_qexecution(
                        node_name=node_name,
                        metric=metric,
                        path_id=path_id,
                        kind=kind,
                        multi=multi,
                        categoryorder=categoryorder,
                        show=True,
                        **style_kwargs,
                    )
                ]
            vpy_plt = PlottingUtils().get_plotting_lib(
                class_name="draw_subplots",
            )[0]
            return vpy_plt.draw_subplots(
                figs=figs,
                rows=rows,
                cols=cols,
                kind=kind,
                subplot_titles=all_metrics,
            )
        node_name = format_type(node_name, dtype=list, na_out=None)
        cond = ""
        if len(node_name) != 0:
            node_name = [f"'{nn}'" for nn in node_name]
            cond = f"node_name IN ({', '.join(node_name)})"
        if not (isinstance(path_id, NoneType)):
            if cond != "":
                cond += " AND "
            cond += f"path_id = {path_id}"
        vdf = self.get_qexecution_report().search(cond)
        if show:
            self._check_vdf_empty(vdf)
            if multi:
                vdf["path_id"].apply("'path_id=' || {}::VARCHAR")
                fun = self._get_chart_method(vdf, kind)
                other_params = {}
                if kind != "pie":
                    other_params = {"kind": "drilldown"}
                return fun(
                    columns=["operator_name", "path_id"],
                    method="sum",
                    of=metric,
                    categoryorder=categoryorder,
                    max_cardinality=1000,
                    **other_params,
                    **style_kwargs,
                )
            else:
                fun = self._get_chart_method(vdf["operator_name"], kind)
                return fun(method="sum", of=metric)
        return vdf[["operator_name", metric]]

    # Step 20: Getting Cluster configuration
    def get_rp_status(self) -> vDataFrame:
        """
        Returns the RP status.

        Returns
        -------
        vDataFrame
            report.

        Examples
        --------

        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        The Cluster Report can also be conveniently
        extracted:

        .. code-block:: python

            qprof.get_rp_status()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cluster_table_2.html

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        query = """SELECT * FROM v_monitor.resource_pool_status;"""
        query = self._replace_schema_in_query(query)
        return vDataFrame(query)

    # Step 21: Getting Cluster configuration
    def get_cluster_config(self) -> vDataFrame:
        """
        Returns the Cluster configuration.

        Returns
        -------
        vDataFrame
            report.

        Examples
        --------

        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        To get cluster configuration details, we can
        conveniently call the function:

        .. code-block:: python

            qprof.get_cluster_config()

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cluster_table.html

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        query = """SELECT * FROM v_monitor.host_resources;"""
        query = self._replace_schema_in_query(query)
        return vDataFrame(query)

    # I/O

    def to_html(self, path: Optional[str] = None) -> str:
        """
        Creates an HTML report.

        Parameters
        ----------
        path: str, optional
            Path where the report will
            be exported.

            .. warning::

                The report will be created in the
                local machine.

        Returns
        -------
        str
            HTML report.

        Examples
        --------

        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can generate easily the HTML report:

        .. code-block:: python

            qprof.to_html()

        .. note::

            For more details, please look at
            :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        cluster_info = self.get_cluster_config()._repr_html_()
        cluster_report = self.get_rp_status()._repr_html_()
        query_execution_report = self.get_qexecution_report()._repr_html_()
        cpu_time_plot = self.get_cpu_time(kind="bar").to_html(full_html=False)
        get_qexecution = self.get_qexecution().to_html(full_html=False)
        get_qsteps = self.get_qsteps(kind="barh").htmlcontent
        get_qplan_profile = self.get_qplan_profile(kind="pie").to_html(full_html=False)
        graphviz_tree = self.get_qplan_tree()
        svg_tree = graphviz_tree.pipe(format="svg").decode("utf-8")
        html_content = f"""
        <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Query Profiling Report</title>
                <!-- Include Plotly resources -->
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <link rel="stylesheet" href="https://cdn.plot.ly/plotly-latest.min.css">
            </head>
            <body>
                <h1>System Configuration</h1>
                <ul>
                    <li>Cluster Information: <br>{cluster_info}</li>
                    <li>Cluster Report: <br>{cluster_report}</li>
                </ul>

                <h1>Query Execution Report</h1>
                {query_execution_report}

                <h1>Execution Time on Each Node</h1>
                <div id="qexecution">{get_qexecution}</div>

                <h1>CPU Time Distribution</h1>
                <div id="cpu_time_distribution">{cpu_time_plot}</div>

                <h1>Query Steps</h1>
                <div id="query_steps">{get_qsteps}</div>

                <h1>Query Plan plot</h1>
                <div id="query_plan_plot">{get_qplan_profile}</div>

                <h1>Query Plan Tree</h1>
                <div id="query_plan_tree">{svg_tree}</div>

            </body>
        </html>
        """

        if path:
            with open(path, "w") as file:
                file.write(html_content)

        return html_content

    # Import Export

    def export_profile(self, filename: os.PathLike) -> None:
        """
        The ``export_profile()`` method provides a high-level
        interface for creating an export bundle of parquet files from a
        QueryProfiler instance.

        The export bundle is a tarball. Inside the tarball there are:
            * ``profile_meta.json``, a file with some information about
            the other files in the tarball
            * Several ``.parquet`` files. There is one ``.parquet`` for
            each system table that
            py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
            uses to analyze query performance.
            * For example, there is a file called ``dc_requests_issued.parquet``.

        Parameters
        --------------
        filename: os.PathLike
            The name of the export bundle to be produced. The input type is
            a synonym for a string or a ``pathlib.Path``.

        Returns
        -------------

        Returns None. Produces ``filename``.

        Examples
        --------

        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler
            from verticapy.performance.vertica.collection.profile_export import ProfileExport

        Now we can profile a query and create a set of system table replicas
        by calling the ``QueryProfiler`` constructor:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;",
                target_schema="replica_001",
                key_id="example123"
            )

        The parameter ``target_schema`` tells the QueryProfiler to create a
        set of replica tables. The parameter ``key_id`` specifies a suffix for all
        of the replica tables associated with this profile. The replica tables are
        a snapshot of the system tables. The replica tables are filtered to contain
        only the information relevant to the query that we have profiled.

        Now we can use ``export_profile`` to produce an export bundle.
        We choose to name our export bundle ``"query_requests_example_001.tar"``.

        .. code-block:: python

            qprof.export_profile(filename="query_requests_example_001.tar")


        After producing an export bundle, we can examine the file contents using
        any tool that read tar-format files. For instance, we can use the tarfile
        library to print the names of all files in the tarball

        .. code-block:: python

            tfile = tarfile.open("query_requests_example_001.tar")
            for f in tfile.getnames():
                print(f"Tarball contains path: {f}")

        The output will be:

        .. code-block::

            Tarball contains path: dc_explain_plans.parquet,
            Tarball contains path: dc_query_executions.parquet
            ...

        """

        if isinstance(self.target_schema, NoneType) or len(self.target_schema) == 0:
            raise ValueError(
                "Export requires that target_schema is set."
                f" Current value of target_schema: {self.target_schema}"
            )

        # Target schema is a dictionary of values
        unique_schemas = set([x for x in self.target_schema.values()])
        if len(unique_schemas) != 1:
            raise ValueError(f"Expected one unique schema, but found {unique_schemas}")
        actual_schema = unique_schemas.pop()
        exp = ProfileExport(
            target_schema=actual_schema, key=self.key_id, filename=filename
        )

        exp.export()

    @staticmethod
    def import_profile(
        target_schema: str,
        key_id: str,
        filename: os.PathLike,
        tmp_dir: os.PathLike = os.getenv("TMPDIR", "/tmp"),
        auto_initialize: bool = True,
    ):
        """
        The static method ``import_profile`` can be used to create new
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler` object
        from the contents of a export bundle.

        Export bundles can be produced
        by the method
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler.export_profile`.
        The bundles contain system table data written into parquet files.

        The method ``import_profile`` executes the following steps:
            * Unpacks the profie bundle
            * Creates tables in the in the target schema if they do not
              exist. The tables will be suffixed by ``key_id``.
            * Copies the data from the parquet files into the tables
            * Creates a ``QueryProfiler`` object initialized to use
              data from the newly created and loaded tables.

        The method returns the new ``QueryProfiler`` object.

        Parameters
        ------------
        target_schema: str
            The name of the schema to load data into
        key_id: str
            The suffix for table names in the target_schema
        filename: os.PathLike
            The file containing exported profile data
        tmp_dir: os.PathLike
            The directory to use for temporary storage of unpacked
            files.


        Returns
        ----------
        A QueryProfiler object


        Examples
        --------

        First, let's import the
        :py:class:`~verticapy.performance.vertica.qprof.QueryProfiler`
        object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler
            from verticapy.performance.vertica.collection.profile_export import ProfileExport

        Now we can profile a query and create a set of system table replicas
        by calling the ``QueryProfiler`` constructor:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;",
                target_schema="replica_001",
                key_id="example123"
            )

        We can use ``export_profile`` to produce an export bundle.
        We choose to name our export bundle ``"query_requests_example_001.tar"``.

        .. code-block:: python

            qprof.export_profile(filename="query_requests_example_001.tar")

        After producing an export bundle, we can import it into a different
        schema using ``import_profile``. For purposes of this example, we'll
        import the data into another schema in the same database. We expect
        it is more common to import the bundle into another database.

        Let's use the import schema name ``import_002``, which is distinct from
        the source schema ``replica_001``.

        .. code-block:: python

            qprof_imported = QueryProfiler.import_profile(
                target_schema="import_002",
                key_id="ex9876",
                filename="query_requests_example_001.tar"
            )

        Now we use the ``QueryProfiler`` to analyze the imported information. All
        ``QueryProfiler`` methods are available. We'll use ``get_qduration()`` as
        an example.

        .. code-block:: python

            print(f"First query duration was {qprof_imported.get_qduration()} seconds")

        Let's assume the query had a duration of 3.14 seconds. The output will be:

        .. code-block::

            First query duration was 3.14 seconds


        """
        pi = ProfileImport(
            # schema and target will be once this test copies
            # files into a schema
            target_schema=target_schema,
            key=key_id,
            filename=filename,
        )
        pi.tmp_path = tmp_dir if isinstance(tmp_dir, Path) else Path(tmp_dir)
        pi.check_schema_and_load_file()
        if auto_initialize:
            return QueryProfiler(target_schema=target_schema, key_id=key_id)
