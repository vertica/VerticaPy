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
from typing import Union


class QprofUtility:
    """
    A class that contains a collection of static methods
    for qprof
    """

    @staticmethod
    def _get_label(row: str, return_path_id: bool = True) -> Union[str, int]:
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
                    return -1001
                if "DELETE" in res:
                    return -1002
                if "UPDATE" in res:
                    return -1003
                if "MERGE" in res:
                    return -1004
                return -1000
            res = res.split("PATH ID: ")[1].split(")")[0]
            res = re.sub(r"[^0-9]", "", res)
            if len(res) == 0:
                return -1
            return int(res)
        return res

    @staticmethod
    def _get_rows(rows: str) -> list:
        """
        ...
        """
        qplan = rows.split("\n")
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
        return [QprofUtility._get_label(row) for row in rows]

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
            "exec_time_ms",
            "est_rows",
            "mem_all_mb",
            "mem_res_mb",
            "proc_rows",
            "prod_rows",
            "pstall_us",
            "rle_prod_rows",
            "rows",
        ]

    @staticmethod
    def _get_metrics_name(metric: str, inv: bool = False) -> str:
        look_up_table = {
            "bytes_spilled": "Number of bytes spilled",
            "clock_time_us": "Clock time in \u00b5s",
            "cost": "Query plan cost",
            "cstall_us": "Network consumer stall time in \u00b5s",
            "exec_time_ms": "Execution time in ms",
            "est_rows": "Estimated row count",
            "mem_res_mb": "Reserved memory size in MB",
            "mem_all_mb": "Allocated memory size in MB",
            "proc_rows": "Processed row count",
            "prod_rows": "Produced row count",
            "pstall_us": "Network producer stall time in \u00b5s",
            "rle_prod_rows": "Produced RLE row count",
            "rows": "Row count",
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
