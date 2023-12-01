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

from verticapy.connection.global_connection import get_global_connection
from verticapy._utils._gen import gen_tmp_name
from verticapy._utils._sql._format import clean_query, quote_ident
from verticapy.connection.connect import current_cursor


def get_dblink_fun(query: str, symbol: str = "$") -> str:
    """
    Returns the SQL needed to deploy the DBLINK UDTF.

    Parameters
    ----------
    query: str
        SQL Query.
    symbol: str
        A special character to identify the connection.
        One of the following:
        "$", "€", "£", "%", "@", "&", "§", "?", "!"

        For example, if the symbol is '$', you can call
        external tables with the input cid by writing
        $$$QUERY$$$, where QUERY represents a custom
        query.

    Returns
    -------
    str
        Formatted SQL Query.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import verticapy.
        import verticapy as vp

        # Set an external connection
        vp.set_external_connection(
            cid = "pgdb",
            rowset = 500,
            symbol = "$",
        )

        # Import the function.
        from verticapy._utils._sql._dblink import get_dblink_fun

        # Generating a query.
        query = "SELECT COUNT(*) FROM postgres_schema.external_table;"

        # Function example.
        get_dblink_fun(query, symbol = "$")

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    gb_conn = get_global_connection()
    external_connections = gb_conn.get_external_connections()
    if symbol not in external_connections:
        raise ConnectionError(
            "External Query detected but no corresponding "
            "Connection Identifier Database is defined (Using "
            f"the symbol '{symbol}'). Use the function connect."
            "set_external_connection to set one with the correct symbol."
        )
    cid = external_connections[symbol]["cid"].replace("'", "''")
    query = query.replace("'", "''")
    rowset = external_connections[symbol]["rowset"]
    query = f"""
        SELECT 
            DBLINK(USING PARAMETERS 
                   cid='{cid}',
                   query='{query}',
                   rowset={rowset}) OVER ()"""
    return clean_query(query)


def replace_external_queries(query: str) -> str:
    """
    Replaces the external queries in the
    input query using the DBLINK UDTF.
    If many external queries are used,
    they are materialised using local
    temporary tables.

    Parameters
    ----------
    query: str
        SQL Query.

    Returns
    -------
    str
        Formatted SQL Query.

    Examples
    --------
    The following code demonstrates
    the usage of the function.

    .. ipython:: python

        # Import verticapy.
        import verticapy as vp

        # Set an external connection
        vp.set_external_connection(
            cid = "pgdb",
            rowset = 500,
            symbol = "$",
        )

        # Import the function.
        from verticapy._utils._sql._dblink import replace_external_queries

        # Generating a query.
        query = "SELECT * FROM $$$postgres_schema.external_table$$$;"

        # Function example.
        replace_external_queries(query, symbol = "$")

    .. note::

        These functions serve as utilities to
        construct others, simplifying the overall
        code.
    """
    gb_conn = get_global_connection()
    external_connections = gb_conn.get_external_connections()
    sql_keyword = (
        "select ",
        "create ",
        "insert ",
        "drop ",
        "backup ",
        "alter ",
        "update ",
    )
    nb_external_queries = 0
    for s in external_connections:
        external_queries = re.findall(f"\\{s}\\{s}\\{s}(.*?)\\{s}\\{s}\\{s}", query)
        for external_query in external_queries:
            if external_query.strip().lower().startswith(sql_keyword):
                external_query_tmp = external_query
                subquery_flag = False
            else:
                external_query_tmp = f"SELECT * FROM {external_query}"
                subquery_flag = True
            query_dblink_template = get_dblink_fun(external_query_tmp, symbol=s)
            if " " in external_query.strip():
                alias = f"VERTICAPY_EXTERNAL_TABLE_{nb_external_queries}"
            else:
                alias = quote_ident(external_query.strip())
            if nb_external_queries >= 1:
                temp_table_name = quote_ident(gen_tmp_name(name=alias))
                create_statement = f"""
                    CREATE LOCAL TEMPORARY TABLE {temp_table_name} 
                    ON COMMIT PRESERVE ROWS 
                    AS {query_dblink_template}"""
                current_cursor().execute(create_statement)
                query_dblink_template = f"v_temp_schema.{temp_table_name} AS {alias}"
            else:
                if subquery_flag:
                    query_dblink_template = f"({query_dblink_template}) AS {alias}"
            query = query.replace(
                f"{s}{s}{s}{external_query}{s}{s}{s}", query_dblink_template
            )
            nb_external_queries += 1
    return query
