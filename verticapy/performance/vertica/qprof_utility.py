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
import re
from typing import Union, Set

from verticapy._utils._sql._sys import _executeSQL


class QprofUtility:
    """
    A class that contains a collection
    of static methods for QPROF.
    """

    @staticmethod
    def _get_label(
        row: str, return_path_id: bool = True, row_idx: int = 0
    ) -> Union[str, int]:
        """
        Gets the label from
        Query Plan chart.

        Parameters
        ----------
        row: str
            Tree row.
        return_path_id: bool, optional
            If set to ``True`` returns
            the path ID instead.
        row_idx: int, optional
            The ID of the row.

        Returns
        -------
        str
            label.

        Examples
        --------
        See :py:meth:`~verticapy.performance.vertica.qprof_utility`
        for more information.
        """
        res = row
        while len(res) > 0 and res[0] in ("+", "-", " ", "|", ">"):
            res = res[1:]
        if return_path_id:
            if "PATH ID: " not in res:
                if "INSERT" in res:
                    return -10000 - row_idx
                if "DELETE" in res:
                    return -20000 - row_idx
                if "UPDATE" in res:
                    return -30000 - row_idx
                if "MERGE" in res:
                    return -40000 - row_idx
                return -1000
            res = res.split("PATH ID: ")[1].split(")")[0]
            mul = 1
            if len(res.strip()) > 0 and "-" == res.strip()[0]:
                mul = -1
            res = re.sub(r"[^0-9]", "", res)
            if len(res) == 0:
                return -1
            return int(res) * mul
        return res

    @staticmethod
    def _get_no_statistics(
        row: str,
    ) -> Union[str, int]:
        """
        Returns if the row
        includes or does not
        include statistics.

        Parameters
        ----------
        row: str
            Tree row.

        Returns
        -------
        bool
            True if the row
            has statistics.

        Examples
        --------
        See :py:meth:`~verticapy.performance.vertica.qprof_utility`
        for more information.
        """
        if "NO STATISTICS" in row:
            return "🚫"
        else:
            return ""

    @staticmethod
    def _get_rows(rows: str) -> list[str]:
        """
        ...
        """
        qplan = rows.split("\n")
        current_id = -1
        for idx, row in enumerate(qplan):
            if (
                "[" in row
                and "]" in row
                and "Cost" in row
                and "Rows" in row
                and "PATH ID: " not in row
            ):
                qplan[idx] += f" (PATH ID: {current_id})"
                current_id -= 1
        n = len(qplan)
        rows_list, tmp_rows = [], []
        for i in range(n):
            if "PATH ID: " in qplan[i] and i > 0:
                rows_list += ["\n".join(tmp_rows)]
                tmp_rows = []
            tmp_rows += [qplan[i]]
        rows_list += ["\n".join(tmp_rows)]
        return rows_list

    @staticmethod
    def _get_path_order(rows: list) -> list:
        """
        ...
        """
        return [
            QprofUtility._get_label(row, row_idx=idx) for idx, row in enumerate(rows)
        ]

    @staticmethod
    def _get_metrics() -> list:
        """
        ...
        """
        return [
            None,
            "bytes_spilled",
            "clock_time_us",
            "cost",
            "cstall_us",
            "exec_time_us",
            "est_rows",
            "mem_all_b",
            "mem_res_b",
            "proc_rows",
            "prod_rows",
            "pstall_us",
            "rle_prod_rows",
            "rows",
            "blocks_filtered_sip",
            "blocks_analyzed_sip",
            "container_rows_filtered_sip",
            "container_rows_filtered_pred",
            "container_rows_pruned_sip",
            "container_rows_pruned_pred",
            "container_rows_pruned_valindex",
            "hash_tables_spilled_sort",
            "join_inner_clock_time_us",
            "join_inner_exec_time_us",
            "join_outer_clock_time_us",
            "join_outer_exec_time_us",
            "network_wait_us",
            "producer_stall_us",
            "producer_wait_us",
            "request_wait_us",
            "response_wait_us",
            "recv_net_time_us",
            "recv_wait_us",
            "rows_filtered_sip",
            "rows_pruned_valindex",
            "rows_processed_sip",
            "total_rows_read_join_sort",
            "total_rows_read_sort",
        ]

    @staticmethod
    def _get_metrics_name(metric: str, inv: bool = False) -> str:
        look_up_table = {
            "bytes_spilled": "Number of bytes spilled",
            "clock_time_us": "AVG Clock time per node in \u00b5s",
            "cost": "Query plan cost",
            "cstall_us": "Network consumer stall time per node in \u00b5s",
            "exec_time_us": "Execution time per node in \u00b5s",
            "est_rows": "Estimated row count",
            "mem_res_b": "Reserved memory size in B",
            "mem_all_b": "Allocated memory size in B",
            "proc_rows": "Processed row count",
            "prod_rows": "Produced row count",
            "pstall_us": "Network producer stall time per node in \u00b5s",
            "rle_prod_rows": "Produced RLE row count",
            "rows": "Row count",
            "blocks_filtered_sip": "Blocks filtered by SIPs expression",
            "blocks_analyzed_sip": "Blocks analyzed by SIPs expression",
            "container_rows_filtered_sip": "Container rows filtered by SIPs expression",
            "container_rows_filtered_pred": "Container rows filtered by query predicates",
            "container_rows_pruned_sip": "Container rows pruned by SIPs expression",
            "container_rows_pruned_pred": "Container rows pruned by query predicates",
            "container_rows_pruned_valindex": "Container rows pruned by valindex",
            "hash_tables_spilled_sort": "Hash tables spilled to sort",
            "join_inner_clock_time_us": "Join inner clock time per node in \u00b5s",
            "join_inner_exec_time_us": "Join inner execution time per node in \u00b5s",
            "join_outer_clock_time_us": "Join outer clock time per node in \u00b5s",
            "join_outer_exec_time_us": "Join outer execution time per node in \u00b5s",
            "network_wait_us": "Network wait time per node in \u00b5s",
            "producer_stall_us": "Producer stall time per node in \u00b5s",
            "producer_wait_us": "Producer wait time per node in \u00b5s",
            "request_wait_us": "Request wait time per node in \u00b5s",
            "response_wait_us": "Response wait time per node in \u00b5s",
            "recv_net_time_us": "Recv net time per node in \u00b5s",
            "recv_wait_us": "Recv wait time per node in \u00b5s",
            "rows_filtered_sip": "Rows filtered by SIPs expression",
            "rows_pruned_valindex": "Rows pruned by valindex",
            "rows_processed_sip": "Rows processed by SIPs expression",
            "total_rows_read_join_sort": "Total rows read in join sort",
            "total_rows_read_sort": "Total rows read in sort",
        }
        if inv:
            look_up_table_inv = {v: k for k, v in look_up_table.items()}
            look_up_table = look_up_table_inv
        if metric in look_up_table:
            return look_up_table[metric]
        return metric

    @staticmethod
    def _get_categoryorder() -> list:
        """
        ...
        """
        return [
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
        ]

    @staticmethod
    def _get_set_of_tables_in_schema(target_schema: str, key: str) -> Set[str]:
        result = _executeSQL(
            f"""SELECT table_name FROM v_catalog.tables 
                    WHERE 
                        table_schema = '{target_schema}'
                        and table_name ilike '%_{key}';
                    """,
            method="fetchall",
        )
        existing_tables = set()
        for row in result:
            existing_tables.add(row[0])
        return existing_tables
