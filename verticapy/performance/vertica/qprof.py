"""
Copyright  (c)  2018-2023 Open Text  or  one  of its
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

    # Step 12
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
        """
        query = """SELECT * FROM v_monitor.host_resources;"""
        return vDataFrame(query)
