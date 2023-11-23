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
import time
from typing import Any, Literal, Optional

import verticapy._config.config as conf
from verticapy.connection.global_connection import get_global_connection
from verticapy._typing import NoneType
from verticapy._utils._sql._dblink import replace_external_queries
from verticapy._utils._sql._display import print_query, print_time
from verticapy._utils._sql._format import (
    clean_query,
    erase_label,
    format_type,
    replace_label,
)
from verticapy.connection.connect import current_cursor


def _executeSQL(
    query: str,
    title: Optional[str] = None,
    data: Optional[list] = None,
    method: Literal[
        "cursor", "fetchrow", "fetchall", "fetchfirstelem", "copy"
    ] = "cursor",
    path: Optional[str] = None,
    print_time_sql: bool = True,
    sql_push_ext: bool = False,
    symbol: str = "$",
) -> Any:
    """
    Executes and returns the result of the input query.
    """
    data = format_type(data, dtype=list)
    special_symbols = get_global_connection().special_symbols
    # Replacing the label
    separator = conf.get_option("label_separator")
    suffix = conf.get_option("label_suffix")
    if isinstance(suffix, NoneType):
        separator = None
    if isinstance(suffix, str) and isinstance(separator, NoneType):
        separator = "__"  # Default separator
    query = replace_label(query, separator=separator, suffix=suffix)
    # Cleaning the query
    if sql_push_ext and (symbol in special_symbols):
        query = erase_label(query)
        query = symbol * 3 + query.replace(symbol * 3, "") + symbol * 3

    elif sql_push_ext and (symbol not in special_symbols):
        raise ValueError(f"Symbol '{symbol}' is not supported.")

    query = replace_external_queries(query)
    query = clean_query(query)

    cursor = current_cursor()
    if conf.get_option("sql_on") and print_time_sql:
        print_query(query, title)
    start_time = time.time()
    if data:
        cursor.executemany(query, data)
    elif method == "copy":
        with open(path, "r", encoding="utf-8") as f:
            cursor.copy(query, f)
    else:
        cursor.execute(query)
    elapsed_time = time.time() - start_time
    if conf.get_option("time_on") and print_time_sql:
        print_time(elapsed_time)
    if method == "fetchrow":
        return cursor.fetchone()
    elif method == "fetchfirstelem":
        return cursor.fetchone()[0]
    elif method == "fetchall":
        return cursor.fetchall()
    return cursor
