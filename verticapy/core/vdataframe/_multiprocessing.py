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
from typing import TYPE_CHECKING

from verticapy._typing import SQLColumns

from verticapy.core.tablesample.base import TableSample

if TYPE_CHECKING:
    from verticapy.core.vdataframe.base import vDataFrame


def aggregate_parallel_block(
    vdf: "vDataFrame", func: list, columns: SQLColumns, ncols_block: int, i: int
) -> TableSample:
    """
    Parallelizes the computations of the aggregate vDataFrame
    method. This allows the vDataFrame to send multiple
    queries at the same time.
    """
    return vdf.aggregate(
        func=func, columns=columns[i : i + ncols_block], ncols_block=ncols_block
    )


def describe_parallel_block(
    vdf: "vDataFrame",
    method: str,
    columns: SQLColumns,
    unique: bool,
    ncols_block: int,
    i: int,
) -> TableSample:
    """
    Parallelizes the computations of the describe vDataFrame
    method. This allows the vDataFrame to send multiple
    queries at the same time.
    """
    return vdf.describe(
        method=method,
        columns=columns[i : i + ncols_block],
        unique=unique,
        ncols_block=ncols_block,
    )
