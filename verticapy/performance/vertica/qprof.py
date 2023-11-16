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
from typing import Any, Literal, Optional

from verticapy.errors import QueryError

from verticapy.core.vdataframe import vDataFrame

from verticapy._typing import NoneType
from verticapy._utils._sql._collect import save_verticapy_logs
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
        resource_pool: Optional[int] = None,
        transaction_id: Optional[int] = None,
        statement_id: Optional[int] = None,
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
            raise NotImplementedError
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

    def _get_interval_str(self, unit: Literal["s", "m", "h"]):
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

    # Step 0
    @staticmethod
    def get_version() -> tuple[int, int, int, int]:
        return vertica_version()

    # Step 1
    def get_request(self, print_request: bool = True) -> str:
        if print_request:
            print(self.request)
        return self.request

    # Step 3
    def get_qsteps(
        self,
        show: bool = True,
        chart_type: Literal[
            "bar",
            "barh",
            "pie",
        ] = "barh",
        unit: Literal["s", "m", "h"] = "s",
    ):
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

    # Step 12
    def get_cpu_time(
        self,
        show: bool = True,
        chart_type: Literal[
            "bar",
            "barh",
        ] = "barh",
    ):
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
        if show:
            fun = self._get_chart_method(vdf, chart_type)
            return fun(
                columns=["node_name", "path_id"], method="SUM(counter_value) AS cet"
            )
        return vdf
