"""
(c)  Copyright  [2018-2023]  OpenText  or one of its
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
from typing import Optional

import vertica_python
from vertica_python.vertica.cursor import Cursor
from vertica_python.vertica.connection import Connection

from verticapy.connection.global_connection import (
    get_global_connection,
    GlobalConnection,
)
from verticapy.connection.read import read_dsn
from verticapy.connection.utils import get_confparser, get_connection_file

"""
Connecting to the DB.
"""


def auto_connect() -> None:
    """
    Automatically creates a connection using the
    auto-connection.
    """
    gb_conn = get_global_connection()
    confparser = get_confparser()

    if confparser.has_section(gb_conn.vpy_auto_connection):
        section = confparser.get(gb_conn.vpy_auto_connection, "name")
    else:
        raise ConnectionError(
            "No Auto Connection available. You can create one using "
            "the 'new_connection' function or set manually a connection"
            " using the 'set_connection' function."
        )
    connect(section)


read_auto_connect = auto_connect


def connect(section: str, dsn: Optional[str] = None) -> None:
    """
    Connects to the database.

    Parameters
    ----------
    section: str
        Name  of the  section in the  configuration
        file.
    dsn: str, optional
        Path to the file containing the credentials.
        If empty, the  Connection File will be used.
    """
    gb_conn = get_global_connection()
    prev_conn = gb_conn.get_connection()
    if not dsn:
        dsn = get_connection_file()
    if prev_conn and not prev_conn.closed():
        prev_conn.close()
    try:
        gb_conn.set_connection(vertica_connection(section, dsn), section, dsn)
    except Exception as e:
        if "The DSN Section" in str(e):
            raise ConnectionError(
                f"The connection '{section}' does not exist. To create "
                "a new connection, you use the 'new_connection' "
                "function with your credentials: {'database': ..., "
                "'host': ..., 'password': ..., 'user': ...}.\n"
                "To view available connections, use the "
                "the 'available_connections' function."
            )
        raise e


def set_connection(conn: Connection) -> None:
    """
    Saves a custom  connection to the VerticaPy object.
    This allows you to specify,  for example, a JDBC or
    ODBC connection. This should not be confused with a
    native   VerticaPy   connection  created   by   the
    'new_connection' function.
    """
    try:
        conn.cursor().execute("SELECT /*+LABEL('connect.set_connection')*/ 1;")
        res = conn.cursor().fetchone()[0]
        assert res == 1
    except Exception as e:
        raise ConnectionError(f"The input connector is not working properly.\n{e}")
    gb_conn = get_global_connection()
    gb_conn.set_connection(conn)


"""
Closing DB Connection.
"""


def close_connection() -> None:
    """
    Closes the connection to the database.
    """
    gb_conn = get_global_connection()
    connection = gb_conn.get_connection()
    if connection and not connection.closed():
        connection.close()


"""
Connections & Cursors Objects.
"""

# Global Connection.


def current_connection() -> GlobalConnection:
    """
    Returns the current  database connection. If the
    connection  is  closed,  VerticaPy  attempts  to
    reconnect with the existing connection.

    If  the  connection   attempt  fails,  VerticaPy
    attempts to reconnect using  stored  credentials.
    If  this  also  fails,   VerticaPy  attempts  to
    connect  using  an  auto  connection.  Otherwise,
    VerticaPy  attempts  to connect to a  VerticaLab
    Environment.
    """
    gb_conn = get_global_connection()
    conn = gb_conn.get_connection()
    dsn = gb_conn.get_dsn()
    section = gb_conn.get_dsn_section()

    # Look if the connection does not exist or is closed

    if not conn or conn.closed():
        # Connection using the existing credentials

        if (section) and (dsn):
            connect(section, dsn)

        else:
            try:
                # Connection using the Auto Connection

                auto_connect()

            except Exception as e:
                try:
                    # Connection to the VerticaLab environment

                    conn = verticapylab_connection()
                    gb_conn.set_connection(conn)

                except:
                    raise e

    return gb_conn.get_connection()


def current_cursor() -> Cursor:
    """
    Returns the current database cursor.
    """
    return current_connection().cursor()


# Local Connection.


def vertica_connection(section: str, dsn: Optional[str] = None) -> Connection:
    """
    Reads the input DSN and creates a Vertica Database
    connection.

    Parameters
    ----------
    section: str
        Name of the section in  the configuration file.
    dsn: str, optional
        Path  to the file containing  the  credentials.
        If empty, the VERTICAPY_CONNECTION environment
        variable will be used.

    Returns
    -------
    conn
        Database connection.
    """
    return vertica_python.connect(**read_dsn(section, dsn))


# VerticaPy Lab Connection.


def verticapylab_connection() -> Connection:
    """
    Returns the VerticaPyLab connection, if possible.

    Returns
    -------
    conn
        Database connection.
    """
    gb_conn = get_global_connection()
    conn_info = {
        "host": "verticadb",
        "port": 5433,
        "user": "dbadmin",
        "password": "",
        "database": "demo",
        "backup_server_node": ["localhost"],
        "session_label": gb_conn.vpy_session_label,
    }
    return vertica_python.connect(**conn_info)
