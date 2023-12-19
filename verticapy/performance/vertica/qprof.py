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
from typing import Any, Callable, Literal, Optional, Union, TYPE_CHECKING
import warnings

from tqdm.auto import tqdm

from verticapy.errors import ExtensionError, QueryError

from verticapy.core.vdataframe import vDataFrame

import verticapy._config.config as conf
from verticapy._typing import NoneType, PlottingObject, SQLExpression
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, format_query, format_type
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import vertica_version

from verticapy.performance.vertica.tree import PerformanceTree
from verticapy.plotting._utils import PlottingUtils

if TYPE_CHECKING and conf.get_import_success("graphviz"):
    from graphviz import Source


class QueryProfiler:
    """
    Base class to profile queries.

    The :py:class:`QueryProfiler` is a valuable tool for
    anyone seeking to comprehend the reasons behind
    a query's lack of performance. It incorporates a
    set of functions inspired by the original QPROF
    project, while introducing an enhanced feature
    set. This includes the capability to generate
    graphics and dashboards, facilitating a
    comprehensive exploration of the data.

    Moreover, it offers greater convenience by
    allowing interaction with an object that
    encompasses various methods and expanded
    possibilities. To initiate the process, all
    that's required is a ``transaction_id`` and a
    ``statement_id``, or simply a query to execute.

    Parameters
    ----------
    .. important::

        :py:class:`QueryProfiler` can only be instantiated with
        either a query or a combination of a transaction
        ID and a statement ID. These parameters cannot be
        both defined and undefined simultaneously.

    request: str, optional
        Query to run.
        The option to run a query is available when
        targeting a query that has not been previously
        executed in the database.

        .. warning::

            It's important to exercise caution; if the
            query is time-consuming, it will require a
            significant amount of time to execute before
            proceeding to the next steps.
    resource_pool: str, optional
        Specify the name of the resource pool to utilize
        when executing the query. Refer to the Vertica
        documentation for a comprehensive list of available
        options.

        .. note::

            This parameter is used only when ``request`` is
            defined.
    transaction_id: int, optional
        ID of the transaction. It refers to a unique
        identifier assigned to a specific transaction
        within the system.
    statement_id: int, optional
        ID of the statement.
    add_profile: bool, optional
        If set to true and the request does not include a
        profile, this option adds the profile keywords at
        the beginning of the query before executing it.

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
        You can ensure this operation by setting
        ``create_copy=True``.
    create_copy: bool, optional
        If set to ``True``, tables or local temporary
        tables will be created by using the schema
        definition of ``target_schema`` parameter
        to store all the Vertica monitor and internal
        meta-tables.

        .. note::

            This parameter is used only when
            ``create_local_temporary_copy=False``.
    create_local_temporary_copy: bool, optional
        If set to ``True``, local temporary tables
        will be created to store all the Vertica
        monitor and internal meta-tables.

        .. note::

            This parameter is used only when
            ``create_copy=False``.
    overwrite: bool
        If set to ``True`` overwrites the
        existing performance tables.

        .. note::

            This parameter is used only when
            ``create_local_temporary_copy=True``.

    Attributes
    ----------
    request: str
        Query.
    transaction_id: int
        Transaction ID.
    statement_id: int
        Statement ID.
    target_schema: dict
        Name of the schema used to store
        all the Vertica monitor and internal
        meta-tables.
    target_tables: dict
        Name of the tables used to store
        all the Vertica monitor and internal
        meta-tables.
    overwrite: bool
        If set to ``True`` overwrites the
        existing performance tables.

    Examples
    --------

    Initialization
    ^^^^^^^^^^^^^^^

    First, let's import the
    :py:class:`QueryProfiler`
    object.

    .. ipython:: python

        from verticapy.performance.vertica import QueryProfiler

    There are multiple ways how we can use the
    :py:class:`QueryProfiler`.

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

        qprof = QueryProfiler(
            transaction_id=45035996273800581,
            statement_id=48,
        )

    .. important::

        To save the different performance tables
        in a specific schema use
        ``target_schema='MYSCHEMA'``, 'MYSCHEMA'
        being the targetted schema. To overwrite
        the tables, use: ``overwrite=True``.
        Finally, if you just need local temporary
        table, use: ``create_local_temporary_copy=True``.

        Example:

        .. code-block:: python

            qprof = QueryProfiler(
                transaction_id=45035996273800581,
                statement_id=48,
                target_schema='MYSCHEMA',
                overwrite=True,
            )

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
            "SELECT * FROM " + titanic["age","fare"].fillna().current_relation())

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

    To avoid recomputing a query, you
    can also directly use a statement
    ID and a transaction ID.

    .. ipython:: python

        qprof = QueryProfiler(
            transaction_id = tid,
            statement_id = sid,
        )

    Accessing the different Performance Tables
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    We can easily look at any Vertica
    Performance Tables easily:

    .. code-block:: python

        qprof.get_table('dc_requests_issued')

    .. ipython:: python
        :suppress:

        result = qprof.get_table('dc_requests_issued')
        html_file = open("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_table_1.html", "w")
        html_file.write(result._repr_html_())
        html_file.close()

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_get_table_1.html

    Executing a QPROF step
    ^^^^^^^^^^^^^^^^^^^^^^^

    Numerous QPROF steps are accessible by directly
    using the corresponding methods. For instance,
    step 0 corresponds to the Vertica version, which
    can be executed using the associated method
    ``get_version``.

    .. ipython:: python

        qprof.get_version()

    .. note::

        To explore all available methods, please
        refer to the 'Methods' section. For
        additional information, you can also
        utilize the ``help`` function.

    It is possible to access the same
    step by using the ``step`` method.

    .. ipython:: python

        qprof.step(idx = 0)

    .. note::

        By changing the ``idx`` value above, you
        can check out all the steps of the
        :py:class:`QueryProfiler`.

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

        qprof.get_qsteps(kind="pie")

    .. ipython:: python
        :suppress:

        import verticapy as vp
        vp.set_option("plotting_lib", "plotly")
        fig = qprof.get_qsteps(kind="pie")
        fig.write_html("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_pie_plot.html")

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
            path_id_info=[5, 6],
            metric='cost',
            shape='square',
            color_low='#0000FF',
            color_high='#FFC0CB',
        )

    .. ipython:: python
        :suppress:

        res = qprof.get_qplan_tree(
            path_id=1,
            path_id_info=[5, 6],
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
        request: Optional[str] = None,
        resource_pool: Optional[str] = None,
        transaction_id: Optional[int] = None,
        statement_id: Optional[int] = None,
        add_profile: bool = True,
        target_schema: Union[None, str, dict] = None,
        create_copy: bool = False,
        create_local_temporary_copy: bool = False,
        overwrite: bool = False,
    ) -> None:
        if create_local_temporary_copy and create_copy:
            raise ValueError(
                "'create_copy' and 'create_local_temporary_copy'"
                " can not be both set to True."
            )
        if create_copy and isinstance(target_schema, NoneType):
            warning_message = (
                "'create_copy' is set to True but 'target_schema' "
                " is empty. The parameters will be both ignored."
            )
            warnings.warn(warning_message, Warning)
        if create_local_temporary_copy and not (isinstance(target_schema, NoneType)):
            warning_message = (
                "'create_local_temporary_copy' is set to True but "
                "'target_schema' is not empty.\nThe parameter "
                "'target_schema' will be ignored."
            )
            warnings.warn(warning_message, Warning)
        if not (isinstance(request, NoneType)) and (
            not (isinstance(transaction_id, NoneType))
            or not (isinstance(statement_id, NoneType))
        ):
            raise ValueError(
                "If the parameter 'request' is defined, you cannot "
                "simultaneously define 'transaction_id' or "
                "'statement_id'."
            )
        elif isinstance(request, NoneType) and (
            isinstance(transaction_id, NoneType) or isinstance(statement_id, NoneType)
        ):
            raise ValueError(
                "Both 'transaction_id' and 'statement_id' must "
                "be defined, or alternatively, the 'request' parameter "
                "must be defined."
            )
        if not (isinstance(request, NoneType)):
            if not (isinstance(resource_pool, NoneType)):
                _executeSQL(
                    f"SET SESSION RESOURCE POOL {resource_pool} ;",
                    title="Setting the resource pool.",
                    method="cursor",
                )
            if add_profile:
                fword = clean_query(request).strip().split()[0].lower()
                if fword != "profile":
                    request = "profile " + request
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
            transaction_id, statement_id = _executeSQL(
                query,
                title="Getting transaction_id, statement_id.",
                method="fetchrow",
            )
            self.request = request

        if not (isinstance(transaction_id, int)):
            raise ValueError(
                "Wrong type for Parameter transaction_id.\n"
                f"Expected integer, found {type(transaction_id)}."
            )
        else:
            self.transaction_id = transaction_id
        if not (isinstance(statement_id, int)):
            raise ValueError(
                "Wrong type for Parameter transaction_id.\n"
                f"Expected integer, found {type(statement_id)}."
            )
        else:
            self.statement_id = statement_id

        # Building the target_schema
        if create_local_temporary_copy:
            self.target_schema = self._v_temp_schema_dict()
            create_table = True
        else:
            if isinstance(target_schema, str):
                self.target_schema = {}
                for schema in self._v_temp_schema_dict():
                    self.target_schema[schema] = target_schema
            else:
                self.target_schema = copy.deepcopy(target_schema)
            create_table = create_copy

        self.overwrite = overwrite
        self._create_copy_v_table(create_table=create_table)

        # Getting the request
        if not (hasattr(self, "request")):
            query = f"""
                SELECT 
                    request 
                FROM v_internal.dc_requests_issued 
                WHERE transaction_id = {transaction_id}
                  AND   statement_id = {statement_id};"""
            query = self._replace_schema_in_query(query)
            try:
                self.request = _executeSQL(
                    query,
                    title="Getting the corresponding query",
                    method="fetchfirstelem",
                )
            except TypeError:
                raise QueryError(
                    f"No transaction with transaction_id={transaction_id} "
                    f"and statement_id={statement_id} was found in the "
                    "v_internal.dc_requests_issued table."
                )

    # Tools

    def _get_interval_str(self, unit: Literal["s", "m", "h"]) -> str:
        unit = str(unit).lower()
        if unit.startswith("s"):
            div = "00:00:01"
        elif unit.startswith("m"):
            div = "00:01:00"
        elif unit.startswith("h"):
            div = "01:00:00"
        else:
            ValueError("Incorrect parameter 'unit'.")
        return div

    def _get_interval(self, unit: Literal["s", "m", "h"]) -> int:
        unit = str(unit).lower()
        if unit.startswith("s"):
            div = 1000000
        elif unit.startswith("m"):
            div = 60000000
        elif unit.startswith("h"):
            div = 3600000000
        else:
            ValueError("Incorrect parameter 'unit'.")
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
            "query_plan_profiles": "v_monitor",
            "query_profiles": "v_monitor",
            "execution_engine_profiles": "v_monitor",
            "resource_pool_status": "v_monitor",
            "host_resources": "v_monitor",
        }

    @staticmethod
    def _v_config_table_list() -> list:
        """
        Config Tables do not use a
        ``transaction_id`` and a
        ``statement_is``.
        """
        return [
            "resource_pool_status",
            "host_resources",
        ]

    def _create_copy_v_table(self, create_table: bool = True) -> None:
        """
        Functions to create a copy
        of the performance tables.
        If the tables exist, it
        will use them to do the
        profiling.
        """
        target_tables = {}
        v_temp_table_dict = self._v_table_dict()
        v_config_table_list = self._v_config_table_list()
        loop = v_temp_table_dict.items()
        if conf.get_option("print_info") and create_table:
            print("Creating a copy of the performance tables...")
        if conf.get_option("tqdm"):
            loop = tqdm(loop, total=len(loop))
        for table, schema in loop:
            sql = "CREATE "
            exists = True
            if (
                not (isinstance(self.target_schema, NoneType))
                and schema in self.target_schema
            ):
                new_schema = self.target_schema[schema]
                new_table = f"{table}_{self.statement_id}_{self.transaction_id}"
                if new_schema == "v_temp_schema":
                    sql += f"LOCAL TEMPORARY TABLE {new_table} ON COMMIT PRESERVE ROWS "
                else:
                    sql += f"TABLE {new_schema}.{new_table}"
                sql += f" AS SELECT * FROM {schema}.{table}"
                if table not in self._v_config_table_list():
                    sql += f" WHERE transaction_id={self.transaction_id} "
                    sql += f"AND statement_id={self.statement_id}"
                target_tables[table] = new_table
                if not (create_table):
                    try:
                        _executeSQL(
                            f"SELECT * FROM {new_schema}.{new_table} LIMIT 0",
                            title="Looking if the relation exists.",
                        )
                    except:
                        exists = False
            if create_table or not (exists):
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

    # Tables

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

             - dc_requests_issued
             - dc_query_executions
             - dc_explain_plans
             - query_plan_profiles
             - query_profiles
             - execution_engine_profiles
             - resource_pool_status
             - host_resources

        Returns
        -------
        vDataFrame
            Vertica DC Table.

        Examples
        --------
        First, let's import the
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        res = format_query(
            query=self.request, indent_sql=indent_sql, print_sql=print_sql
        )
        if return_html:
            return res[1]
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        query = f"""
            SELECT
                query_duration_us 
            FROM 
                v_monitor.query_profiles 
            WHERE 
                transaction_id={self.transaction_id} AND 
                statement_id={self.statement_id};"""
        query = self._replace_schema_in_query(query)
        qd = _executeSQL(
            query,
            title="Getting the corresponding query",
            method="fetchfirstelem",
        )
        return float(qd / self._get_interval(unit))

    # Step 3: Query execution steps
    def get_qsteps(
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
        :py:class:`QueryProfiler`
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

            qprof.get_qsteps(kind="pie")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_pie_plot.html

        .. note::

            For more details, please look at
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        div = self._get_interval_str(unit)
        query = f"""
            SELECT
                execution_step, 
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
            fun = self._get_chart_method(vdf["execution_step"], kind)
            return fun(
                method="max",
                of="elapsed",
                categoryorder=categoryorder,
                max_cardinality=1000,
                **style_kwargs,
            )
        return vdf

    # Step 5: Query plan
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
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
        metric: Literal[None, "cost", "rows"] = "rows",
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
        metric: str, optional
            The metric used to color
            the tree nodes. One of
            the following:

            - cost
            - rows
            - None (no specific color)

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
                Default: #000000 (black)
            - fillcolor:
                Color used to fill the
                nodes in case no gradient
                is computed: ``metric=None``.
                Default: #ADD8E6 (lightblue)
            - edge_color:
                Edge color.
                Default: #000000 (black)
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
            - info_fontsize
                Information box font
                size.
                Default: 8

        Returns
        -------
        graphviz.Source
            graphviz object.

        Examples
        --------
        First, let's import the
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        rows = self.get_qplan(print_plan=False)
        obj = PerformanceTree(
            rows,
            show_ancestors=show_ancestors,
            path_id_info=path_id_info,
            path_id=path_id,
            metric=metric,
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
        """
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
            WHERE transaction_id = {self.transaction_id}{where}
            ORDER BY
                statement_id,
                path_id,
                path_line_index;"""
        query = self._replace_schema_in_query(query)
        vdf = vDataFrame(query).sort(["stmtid", "path_id", "path_line_index"])
        if show:
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
        :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        query = f"""
            SELECT 
                node_name, 
                path_id::VARCHAR, 
                counter_value
            FROM 
                v_monitor.execution_engine_profiles 
            WHERE 
                counter_name = 'execution time (us)' AND 
                transaction_id={self.transaction_id} AND 
                statement_id={self.statement_id}"""
        query = self._replace_schema_in_query(query)
        vdf = vDataFrame(query)
        columns = ["path_id", "node_name"]
        if reverse:
            columns.reverse()
        if show:
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        query = f"""
            SELECT
                node_name,
                operator_name,
                path_id,
                ROUND(SUM(CASE counter_name WHEN 'execution time (us)' THEN
                    counter_value ELSE NULL END) / 1000, 3.0) AS exec_time_ms,
                SUM(CASE counter_name WHEN 'estimated rows produced' THEN
                    counter_value ELSE NULL END) AS est_rows,
                SUM(CASE counter_name WHEN 'rows processed' THEN
                    counter_value ELSE NULL END) AS proc_rows,
                SUM(CASE counter_name WHEN 'rows produced' THEN
                    counter_value ELSE NULL END) AS prod_rows,
                SUM(CASE counter_name WHEN 'rle rows produced' THEN
                    counter_value ELSE NULL END) AS rle_prod_rows,
                SUM(CASE counter_name WHEN 'consumer stall (us)' THEN
                    counter_value ELSE NULL END) AS cstall_us,
                SUM(CASE counter_name WHEN 'producer stall (us)' THEN
                    counter_value ELSE NULL END) AS pstall_us,
                ROUND(SUM(CASE counter_name WHEN 'memory reserved (bytes)' THEN
                    counter_value ELSE NULL END)/1000000, 1.0) AS mem_res_mb,
                ROUND(SUM(CASE counter_name WHEN 'memory allocated (bytes)' THEN 
                    counter_value ELSE NULL END) / 1000000, 1.0) AS mem_all_mb
            FROM
                v_monitor.execution_engine_profiles
            WHERE
                transaction_id={self.transaction_id} AND
                statement_id={self.statement_id} AND
                counter_value / 1000000 > 0
            GROUP BY
                1, 2, 3
            ORDER BY
                CASE WHEN SUM(CASE counter_name WHEN 'execution time (us)' THEN
                    counter_value ELSE NULL END) IS NULL THEN 1 ELSE 0 END ASC,
                5 DESC;"""
        query = self._replace_schema_in_query(query)
        return vDataFrame(query)

    # Step 14B: Query execution chart
    def get_qexecution(
        self,
        node_name: Union[None, str, list] = None,
        metric: Literal[
            "all",
            "exec_time_ms",
            "est_rows",
            "proc_rows",
            "prod_rows",
            "rle_prod_rows",
            "cstall_us",
            "pstall_us",
            "mem_res_mb",
            "mem_all_mb",
        ] = "exec_time_ms",
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
             - exec_time_ms (default)
             - est_rows
             - proc_rows
             - prod_rows
             - rle_prod_rows
             - cstall_us
             - pstall_us
             - mem_res_mb
             - mem_all_mb
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        if metric == "all" and show:
            if conf.get_option("plotting_lib") != "plotly":
                raise ExtensionError(
                    "Plots with metric='all' is only available for Plotly Integration."
                )
            figs = []
            all_metrics = [
                "exec_time_ms",
                "est_rows",
                "proc_rows",
                "prod_rows",
                "rle_prod_rows",
                "cstall_us",
                "pstall_us",
                "mem_res_mb",
                "mem_all_mb",
            ]
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
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
        :py:class:`QueryProfiler`
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
            :py:class:`verticapy.performance.vertica.qprof.QueryProfiler`.
        """
        query = """SELECT * FROM v_monitor.host_resources;"""
        query = self._replace_schema_in_query(query)
        return vDataFrame(query)
