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
    _clean_query: bool = True,
) -> Any:
    """
    Executes and returns the
    result of the input query.

    Parameters
    ----------
    query: str
        SQL Query.
    title: str, optional
        Query Title. It will be displayed
        with the query if the option is
        activated.
    data: list, optional
        It represents the data we want to
        ingest in the database.
    method: str, optional
        Method to use when executing the
        query.

         - cursor:
            Executes the query and returns
            the cursor.
         - fetchrow:
            Executes the query and returns
            the first row.
         - fetchall:
            Executes the query and returns
            the entire result.
         - fetchfirstelem:
            Executes the query and returns
            the first element.
         - copy:
            Ingests the data and returns
            the cursor.
    path: str, optional
        Only used when ``method = 'copy'``.
        Path to the file to ingest.
    print_time_sql: bool, optional
        If set to ``True``, and the associated
        option is activated, it prints the
        SQL query and the time of execution.
    sql_push_ext: bool, optional
        If set to ``True``, DBLINK
        is activated.
    symbol: str, optional
        A special character to identify the
        connection. One of the following:
        ``"$", "€", "£", "%", "@", "&", "§", "?", "!"``

        For example, if the symbol is '$',
        you can call external tables with
        the input cid by writing $$$QUERY$$$,
        where QUERY represents a custom
        query.

    Returns
    -------
    The result depends on the input method;
    it can be the cursor or the result of
    the query.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import the function.
        from verticapy._utils._sql._sys import _executeSQL

        # Generating a SQL query.
        query = "SELECT 1, 2, 3"

        # Executing the query and returning the cursor.
        _executeSQL(query, method = 'cursor')

        # Executing the query and returning the first row.
        _executeSQL(query, method = 'fetchrow')

        # Executing the query and returning the first element.
        _executeSQL(query, method = 'fetchfirstelem')

        # Executing the query and returning the entire result.
        _executeSQL(query, method = 'fetchall')

    .. note::

        In case of insertion or ingestion, you
        can use the 'copy' ``method`` to simplify
        the process.

    .. note::

        This function is one of the most crucial
        functions in the entire API. All queries
        pass through it before being executed.

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
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

    if _clean_query:
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
