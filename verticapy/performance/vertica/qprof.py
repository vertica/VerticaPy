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
from typing import Any, Literal, Optional, Union

from verticapy.errors import QueryError

from verticapy.core.vdataframe import vDataFrame

from verticapy._typing import NoneType, PlottingObject
from verticapy._utils._sql._collect import save_verticapy_logs
from verticapy._utils._sql._format import clean_query, format_query
from verticapy._utils._sql._sys import _executeSQL
from verticapy._utils._sql._vertica_version import vertica_version


class QueryProfiler:
    """
    Base class to profile queries.

    The ``QueryProfiler`` is a valuable tool for anyone seeking
    to comprehend the reasons behind a query's lack of
    performance. It incorporates a set of functions inspired
    by the original QPROF project, while introducing an enhanced
    feature set. This includes the capability to generate graphics
    and dashboards, facilitating a comprehensive exploration of
    the data.

    Moreover, it offers greater convenience by allowing
    interaction with an object that encompasses various methods
    and expanded possibilities. To initiate the process, all that's
    required is a transaction_id and a statement_id, or simply a
    query to execute.

    Parameters
    ----------
    .. important::

        QueryProfiler can only be instantiated with either a query
        or a combination of a transaction ID and a statement ID.
        These parameters cannot be both defined and undefined
        simultaneously.

    request: str, optional
        Query to run.
        The option to run a query is available when targeting a query
        that has not been previously executed in the database.

        .. warning::

            It's important to exercise caution; if the query is
            time-consuming, it will require a significant amount
            of time to execute before proceeding to the next steps.
    resource_pool: str, optional
        Specify the name of the resource pool to utilize when executing
        the query. Refer to the Vertica documentation for a comprehensive
        list of available options.

        .. note::

            This parameter is used only when 'request' is defined.
    transaction_id: int, optional
        ID of the transaction. It refers to a unique identifier assigned
        to a specific transaction within the system.
    statement_id: int, optional
        ID of the statement.
    add_profile: bool, optional
        If set to true and the request does not include a profile, this
        option adds the profile keywords at the beginning of the query
        before executing it.

        .. note::

            This parameter is used only when 'request' is defined.

    Attributes
    ----------
    request: str
        Query.
    transaction_id: int
        Transaction ID.
    statement_id: int
        Statement ID.

    Examples
    --------

    Initialization
    ^^^^^^^^^^^^^^^

    First, let's import the QueryProfiler object.

    .. ipython:: python

        from verticapy.performance.vertica import QueryProfiler

    There are multiple ways how we can use the Query Profiler.

    - From ``transaction_id`` and ``statement_id``
    - From SQL generated from verticapy functions
    - Directly from SQL query

    **Transaction ID and Statement ID**

    In this example, we run a groupby command on the
    amazon dataset.

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

        Above we use the ``WHERE`` command in order to filter
        only those results that match our query above.
        You can use these filters to sift through the list
        of queries.

    Once we have the ``transaction_id`` and ``statement_id``
    we can directly use it:

    .. code-block:: python

        qprof = QueryProfiler(
            transaction_id=45035996273800581,
            statement_id=48,
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

    The query is then executed, and you can easily retrieve the
    statement and transaction IDs.

    .. ipython:: python

        tid = qprof.transaction_id
        sid = qprof.statement_id
        print(f"tid={tid};sid={sid}")

    To avoid recomputing a query, you can also directly use a
    statement ID and a transaction ID.

    .. ipython:: python

        qprof = QueryProfiler(transaction_id=tid, statement_id=sid)

    Executing a QPROF step
    ^^^^^^^^^^^^^^^^^^^^^^^

    Numerous QPROF steps are accessible by directly using the corresponding
    methods. For instance, step 0 corresponds to the Vertica version, which
    can be executed using the associated method ``get_version``.

    .. ipython:: python

        qprof.get_version()

    .. note::

        To explore all available methods, please refer to the 'Methods'
        section. For additional information, you can also utilize the
        ``help`` function.

    It is possible to access the same step by using the ``step`` method.

    .. ipython:: python

        qprof.step(idx=0)

    .. note::

        By changing the ``idx`` value above, you
        can check out all the steps of the Query Profiler.

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

        You can change the unit to "m" to get the
        result in minutes.

    **Query Execution Time Plots**

    To get the time breakdown of all the
    steps in a graphical output, we can call
    the ``get_qsteps`` attribute.

    .. code-block:: python

        qprof.get_qsteps(chart_type="pie")

    .. ipython:: python

        import verticapy as vp
        vp.set_option("plotting_lib", "plotly")
        fig = qprof.get_qsteps(chart_type="pie")
        fig.write_html("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_pie_plot.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_pie_plot.html

    .. note::

        The same plot can also be plotted using
        a bar plot by switching the ``chart_type``
        to "bar".

    **Query Plan**

    To get the entire query plan:

    .. ipython:: python

        qprof.get_qplan()

    **Query Plan Profile**

    To visualize the time consumption of
    query profile plan:

    .. code-block:: python

        qprof.get_qplan_profile(chart_type="pie")

    .. ipython:: python

        fig = qprof.get_qplan_profile(chart_type="pie")
        fig.write_html("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_qplan_profile.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_qplan_profile.html

    .. note::

        The same plot can also be plotted using
        a bar plot by switching the ``chart_type``
        to "bar".

    **CPU Time by Node and Path ID**

    Another very important metric could be the CPU time
    spent by each node. This can be visualized by:

    .. code-block:: python

        qprof.get_cpu_time(chart_type="bar")

    .. ipython:: python

        fig = qprof.get_qplan_profile(chart_type="pie")
        fig.write_html("SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cup_node.html")

    .. raw:: html
        :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cup_node.html

    In order to get the results in a tabular form,
    just switch the ``show`` option to ``False''.

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

    To get node-wise performance infomation,
    ``get_qexecution`` can be used:

    .. code-block:: python

        qprof.get_qexecution(
            node_name="v_vdash_node0003",
            metric="exec_time_ms",
            chart_type="pie",
        )

    .. note::

        The node name is different for different
        configurations. You can search for the node
        names in the full report.

    **Cluster**

    To get cluster configuration details, we can
    use:

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

    The Cluster Report can also be conveniently
    extracted:

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

        Each method may have multiple parameters and options. It is
        essential to refer to the documentation of each method to
        understand its usage.
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
    ) -> None:
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
        if not (hasattr(self, "request")):
            query = f"""
                SELECT 
                    request 
                FROM v_internal.dc_requests_issued 
                WHERE transaction_id = {transaction_id}
                  AND   statement_id = {statement_id};"""
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
        chart_type: Literal[
            "bar",
            "barh",
            "pie",
        ],
    ):
        chart_type = str(chart_type).lower()
        if chart_type == "pie":
            return v_object.pie
        elif chart_type == "bar":
            return v_object.bar
        elif chart_type == "barh":
            return v_object.barh
        else:
            ValueError("Incorrect parameter 'chart_type'.")

    # Main Method

    def step(self, idx: int, *args, **kwargs) -> Any:
        steps_id = {
            0: self.get_version,
            1: self.get_request,
            2: self.get_qduration,
            3: self.get_qsteps,
            4: NotImplemented,
            5: self.get_qplan,
            6: self.get_qplan_profile,
            7: NotImplemented,
            8: NotImplemented,
            9: NotImplemented,
            10: NotImplemented,
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
            MAJOR, MINOR, PATCH, POST

        Examples
        ---------

        First, let's import the QueryProfiler object.

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
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
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
        ---------

        First, let's import the QueryProfiler object.

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
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
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
        ---------

        First, let's import the QueryProfiler object.

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
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
        """
        query = f"""
            SELECT
                query_duration_us 
            FROM 
                v_monitor.query_profiles 
            WHERE 
                transaction_id={self.transaction_id} AND 
                statement_id={self.statement_id};"""
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
        chart_type: Literal[
            "bar",
            "barh",
            "pie",
        ] = "pie",
        show: bool = True,
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

        chart_type: str, optional
            Chart Type.

            - bar:
                Bar Chart.

            - barh:
                Horizontal Bar Chart.

            - pie:
                Pie Chart.

        show: bool, optional
            If set to True, the Plotting object
            is returned.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        First, let's import the QueryProfiler object.

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

            qprof.get_qsteps(chart_type="pie")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_pie_plot.html

        .. note::

            For more details, please look at
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
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
        vdf = vDataFrame(query)
        if show:
            fun = self._get_chart_method(vdf["execution_step"], chart_type)
            return fun(method="max", of="elapsed")
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
            If set to True, the query plan
            report is returned.
        print_plan: bool, optional
            If set to True, the query plan
            is printed.

        Returns
        -------
        str
            Query Plan.

        Examples
        ---------

        First, let's import the QueryProfiler object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        We can easily call the function to get the entire query plan:

            .. ipython:: python

                qprof.get_qplan()

        .. note::

            For more details, please look at
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
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
            ORDER BY 
                statement_id,
                path_id,
                path_line_index;"""
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

    # Step 6: Query plan profile
    def get_qplan_profile(
        self,
        unit: Literal["s", "m", "h"] = "s",
        chart_type: Literal[
            "bar",
            "barh",
            "pie",
        ] = "pie",
        show: bool = True,
    ) -> Union[PlottingObject, vDataFrame]:
        """
        Returns the Query Plan chart.

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
        chart_type: str, optional
            Chart Type.

            - bar:
                Bar Chart.

            - barh:
                Horizontal Bar Chart.

            - pie:
                Pie Chart.

        show: bool, optional
            If set to True, the Plotting object
            is returned.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        First, let's import the QueryProfiler object.

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

            qprof.get_qplan_profile(chart_type="pie")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_qplan_profile.html

        .. note::

            For more details, please look at
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
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
        vdf = vDataFrame(query).sort(["stmtid", "path_id", "path_line_index"])
        if show:
            fun = self._get_chart_method(vdf["path_line"], chart_type)
            return fun(method="sum", of="running_time")
        return vdf

    # Step 12: CPU Time by node and path_id
    def get_cpu_time(
        self,
        chart_type: Literal[
            "bar",
            "barh",
        ] = "bar",
        reverse: bool = False,
        show: bool = True,
    ) -> Union[PlottingObject, vDataFrame]:
        """
        Returns the CPU Time by node and path_id chart.

        Parameters
        ----------
        chart_type: str, optional
            Chart Type.

            - bar:
                Bar Chart.

            - barh:
                Horizontal Bar Chart.
        reverse: bool, optional
            If set to True, the Plotting object
            is returned.
        show: bool, optional
            If set to True, the Plotting object
            is returned.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        First, let's import the QueryProfiler object.

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

            qprof.get_cpu_time(chart_type="bar")

        .. raw:: html
            :file: SPHINX_DIRECTORY/figures/performance_vertica_query_profiler_cup_node.html

        .. note::

            For more details, please look at
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
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
        vdf = vDataFrame(query)
        columns = ["path_id", "node_name"]
        if reverse:
            columns.reverse()
        if show:
            fun = self._get_chart_method(vdf, chart_type)
            return fun(columns=columns, method="SUM(counter_value) AS cet")
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
        ---------

        First, let's import the QueryProfiler object.

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
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
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
        return vDataFrame(query)

    # Step 14B: Query execution chart
    def get_qexecution(
        self,
        node_name: str,
        metric: Literal[
            "exec_time_ms",
            "est_rows",
            "proc_rows",
            "prod_rows",
            "rle_prod_rows",
            "cstall_us",
            "pstall_us",
            "mem_res_mb",
            "mem_all_mb",
        ],
        path_id: Optional[int] = None,
        chart_type: Literal[
            "bar",
            "barh",
            "pie",
        ] = "barh",
        show: bool = True,
    ) -> Union[PlottingObject, vDataFrame]:
        """
        Returns the Query execution chart.

        Parameters
        ----------
        node_name: str
            Node name.
        metric: str
            Metric to use. One of the following:
                - exec_time_ms
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
        chart_type: str, optional
            Chart Type.

            - bar:
                Bar Chart.

            - barh:
                Horizontal Bar Chart.

            - pie:
                Pie Chart.

        show: bool, optional
            If set to True, the Plotting object
            is returned.

        Returns
        -------
        obj
            Plotting Object.

        Examples
        ---------

        First, let's import the QueryProfiler object.

        .. code-block:: python

            from verticapy.performance.vertica import QueryProfiler

        Then we can create a query:

        .. code-block:: python

            qprof = QueryProfiler(
                "select transaction_id, statement_id, request, request_duration"
                " from query_requests where start_timestamp > now() - interval'1 hour'"
                " order by request_duration desc limit 10;"
            )

        To get node-wise performance infomation,
        ``get_qexecution`` can be used:

        .. code-block:: python

            qprof.get_qexecution(
                node_name="v_vdash_node0003",
                metric="exec_time_ms",
                chart_type="pie",
            )

        .. note::

            The node name is different for different
            configurations. You can search for the node
            names in the full report.

        .. note::

            For more details, please look at
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
        """
        cond = f"node_name = '{node_name}'"
        if not (isinstance(path_id, NoneType)):
            cond += f" AND path_id = {path_id}"
        vdf = self.get_qexecution_report().search(cond)
        if show:
            fun = self._get_chart_method(vdf["operator_name"], chart_type)
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
        ---------

        First, let's import the QueryProfiler object.

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
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
        """
        query = """SELECT * FROM v_monitor.resource_pool_status;"""
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
        ---------

        First, let's import the QueryProfiler object.

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
            :py:class:`verticapy.performance.vertica.QueryProfiler`.
        """
        query = """SELECT * FROM v_monitor.host_resources;"""
        return vDataFrame(query)
