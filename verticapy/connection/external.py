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
from verticapy.connection.global_connection import get_global_connection


def set_external_connection(cid: str, rowset: int = 500, symbol: str = "$") -> None:
    """
    Sets a Connection Identifier Database. It connects to
    an external source using DBLINK. For more information,
    see: https://github.com/vertica/dblink

    Parameters
    ----------
    cid: str
        Connection Identifier Database.
    rowset: int, optional
        Number of rows retrieved from the remote database
        during each SQLFetch() cycle.
    symbol: str, optional
        A special character to identify the connection.
        One of the following:
        "$", "€", "£", "%", "@", "&", "§", "?", "!"

        For example, if the symbol is '$', you can call
        external tables with the input cid by writing
        $$$QUERY$$$, where QUERY represents a custom
        query.

    Example
    -------
    Set up a connection with a database using the alias "pgdb"

    .. note:: When configuring an external connection, you'll need to assign a unique symbol to identify it. This symbol will subsequently allow you to extract data from the target database using the associated identifier.

    .. code-block:: python

        import verticapy as vp

        vp.set_external_connection(
            cid = "pgdb",
            rowset = 500,
            symbol = "&")
    """
    gb_conn = get_global_connection()
    gb_conn.set_external_connections(symbol, cid, rowset)
