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

from typing import Optional

import vertica_python
from vertica_python.vertica.cursor import Cursor
from vertica_python.vertica.connection import Connection

import verticapy._config.config as conf
from verticapy.connection.errors import ConnectionError, OAuthTokenRefreshError
from verticapy.connection.global_connection import (
    get_global_connection,
    GlobalConnection,
)
from verticapy.connection.read import read_dsn
from verticapy.connection.utils import get_confparser, get_connection_file
from verticapy.connection.write import new_connection
from verticapy.connection.oauth_manager import OAuthManager

"""
Connecting to the DB.
"""


def auto_connect() -> None:
    """
    Automatically creates
    a connection using the
    auto-connection.

    Examples
    --------
    Connects using an existing
    auto-connection:

    .. code-block:: python

        from verticapy.connection import auto_connect

        auto_connect()

    .. seealso::

        | :py:func:`~verticapy.connection.available_connections` :
            Displays all available connections.
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
        Name of the section in the
        configuration file.
    dsn: str, optional
        Path to the file containing
        the credentials. If empty,
        the Connection File will be
        used.

    Examples
    --------
    Display all available connections:

    .. code-block:: python

        from verticapy.connection import available_connections

        available_connections()

    ``['VML', 'VerticaDSN', 'VerticaDSN_test']``

    Connect using the VerticaDSN connection:

    .. code-block:: python

        from verticapy.connection import connect

        connect("VerticaDSN")

    .. seealso::

        | :py:func:`~verticapy.connection.available_connections` :
            Displays all available connections.
        | :py:func:`~verticapy.connection.get_connection_file` :
            Gets the VerticaPy connection file.
        | :py:func:`~verticapy.connection.new_connection` :
            Creates a new VerticaPy connection.
        | :py:func:`~verticapy.connection.set_connection` :
            Sets the VerticaPy connection.
    """
    gb_conn = get_global_connection()
    prev_conn = gb_conn.get_connection()
    if not dsn:
        dsn = get_connection_file()
    if prev_conn and not prev_conn.closed():
        prev_conn.close()
    try:
        connection_config = read_dsn(section, dsn)
        # if the user has provided a refresh token, do token refresh, update the config's oauth access token
        if connection_config.get("oauth_refresh_token", False):
            oauth_manager = OAuthManager(connection_config["oauth_refresh_token"])
            oauth_manager.set_config(connection_config["oauth_config"])
            connection_config["oauth_access_token"] = oauth_manager.do_token_refresh()
            gb_conn.set_connection(
                vertica_connection(section=None, dsn=None, config=connection_config)
            )
        else:
            gb_conn.set_connection(
                vertica_connection(section, dsn, config=None), section, dsn
            )
        if conf.get_option("print_info"):
            print("Connected Successfully!")
    except OAuthTokenRefreshError as error:
        print(
            "Access Denied: Your authentication credentials are incorrect or have expired. Please retry"
        )
        new_connection(
            conn_info=read_dsn(section, dsn), prompt=True, connect_attempt=False
        )
        try:
            gb_conn.set_connection(
                vertica_connection(section, dsn, config=None), section, dsn
            )
            if conf.get_option("print_info"):
                print("Connected Successfully!")
        except OAuthTokenRefreshError as error:
            print("Error persists:")
            raise error
    except ConnectionError as error:
        print(
            "A connection error occured. Common reasons may be an invalid host, port, or, if requiring "
            "OAuth and token refresh, this may be due to an incorrect or malformed token url."
        )
        raise error
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
    Saves a custom connection to the
    VerticaPy object. This allows you
    to specify, for example, a JDBC or
    ODBC connection. This should not be
    confused with a native VerticaPy
    connection created by the
    :py:func:`~verticapy.connection.new_connection`
    function.

    Examples
    --------
    Create a connection using the
    official Vertica Python client:

    .. note::

        You can use any connector (ODBC,
        JDBC, etc.) as long it has both
        ``fetchone`` and ``fetchall``
        methods. However, note that
        VerticaPy works most efficiently
        with the native client because
        of its support for various complex
        data types and certain Vertica
        optimizations.

    .. code-block:: python

        import vertica_python

        conn_info = {
            'host': "10.211.55.14",
            'port': 5433,
            'user': "dbadmin",
            'password': "XxX",
            'database': "testdb",
        }
        conn = vertica_python.connect(** conn_info)

    Set up the connector:

    .. warning::

        As this connector is used
        throughout the entire API,
        if it's closed, you'll need
        to create a new one. This is
        why, in some cases, it's better
        to use auto-connection, which
        automatically create a new
        connection if the current one
        is closed.

    .. code-block:: python

        from verticapy.connection import set_connection

        set_connection(conn)

    .. seealso::

        | :py:func:`~verticapy.connection.new_connection` :
            Creates a new VerticaPy connection.
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
    Closes the connection
    to the database.

    Examples
    --------
    Close all current connections:

    .. warning::

        When you close the connection,
        your session will terminate and
        all temporary elements will be
        automatically dropped.

    .. code-block:: python

        from verticapy.connection import close_connection

        close_connection()

    .. seealso::

        | :py:func:`~verticapy.connection.current_connection` :
            Returns the current VerticaPy connection.
        | :py:func:`~verticapy.connection.set_connection` :
            Sets the VerticaPy connection.
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
    Returns the current database
    connection. If the connection
    is closed, VerticaPy attempts
    to reconnect with the existing
    connection.

    If the connection attempt fails,
    VerticaPy attempts to reconnect
    using stored credentials.
    If this also fails, VerticaPy
    attempts to connect using an
    auto connection. Otherwise,
    VerticaPy attempts to connect
    to a VerticaLab Environment.

    Examples
    --------
    Get the current VerticaPy connection:

    .. code-block:: python

        from verticapy.connection import current_connection

        conn = current_connection()
        conn

    ``<vertica_python.vertica.connection.Connection at 0x118c1f8d0>``

    After the connection is
    established, you can execute
    SQL queries directly:

    .. note::

        Please refer to your connector's
        API reference for a comprehensive
        list of its functionalities.

    .. code-block:: python

        conn.cursor().execute("SELECT version();").fetchone()

    ``['Vertica Analytic Database v12.0.4-0']``

    .. seealso::

        | :py:func:`~verticapy.connection.current_cursor` :
            Returns the current VerticaPy cursor.
        | :py:func:`~verticapy.connection.new_connection` :
            Creates a new VerticaPy connection.
        | :py:func:`~verticapy.connection.set_connection` :
            Sets the VerticaPy connection.
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
    Returns the current
    database cursor.

    Examples
    --------
    Get the current cursor:

    .. code-block:: python

        from verticapy.connection import current_cursor

        cur = current_cursor()
        cur

    ``<vertica_python.vertica.cursor.Cursor at 0x11a7b4748>``

    Directly execute an SQL query:

    .. code-block:: python

        cur.execute("SELECT version();").fetchone()

    ``['Vertica Analytic Database v12.0.4-0']``

    .. seealso::

        | :py:func:`~verticapy.connection.current_connection` :
            Returns the current VerticaPy connection.
        | :py:func:`~verticapy.connection.new_connection` :
            Creates a new VerticaPy connection.
        | :py:func:`~verticapy.connection.set_connection` :
            Sets the VerticaPy connection.
    """
    return current_connection().cursor()


# Local Connection.


def vertica_connection(
    section: Optional[str] = None,
    dsn: Optional[str] = None,
    config: Optional[dict] = None,
) -> Connection:
    """
    Reads the input DSN and
    creates a Vertica Database
    connection.

    Parameters
    ----------
    section: str, optional
        Name of the section in
        the configuration file.
    dsn: str, optional
        Path to the file containing
        the credentials. If empty,
        the ``VERTICAPY_CONNECTION``
        environment variable will
        be used.
    config: dict, optional
        Configuration object override
        used to create a connection.
        If specified, will ignore the
        section and dsn properties.

    Returns
    -------
    conn
        Database connection.

    Examples
    --------
    Create a connection using the input DSN:

    .. note::

        This example utilizes a Data
        Source Name (DSN) to establish
        the connection, which is stored
        in the file specified by the
        global variable ``VERTICAPY_CONNECTION``.
        However, if you prefer a customized
        file with a different location, you
        can specify the file path accordingly.

    .. code-block:: python

        from verticapy.connection import vertica_connection

        vertica_connection("VerticaDSN")

    ``<vertica_python.vertica.connection.Connection at 0x106526198>``

    .. seealso::

        | :py:func:`~verticapy.connection.current_connection` :
            Returns the current VerticaPy connection.
        | :py:func:`~verticapy.connection.new_connection` :
            Creates a new VerticaPy connection.
        | :py:func:`~verticapy.connection.set_connection` :
            Sets the VerticaPy connection.
    """
    return (
        vertica_python.connect(**config)
        if config
        else vertica_python.connect(**read_dsn(section, dsn))
    )


# VerticaPy Lab Connection.


def verticapylab_connection() -> Connection:
    """
    Returns the VerticaPyLab
    connection, if possible.

    Returns
    -------
    conn
        Database connection.

    Examples
    --------
    Get the VerticaPyLab connection:

    .. note::

        VerticaPyLab is a Dockerized
        environment designed for
        seamlessly using VerticaPy.
        This function returns the
        connection to the Vertica
        instance within the lab,
        allowing for necessary
        environment customization.

    .. code-block:: python

        from verticapy.connection import verticalab_connection

        verticalab_connection()

    ``<vertica_python.vertica.connection.Connection at 0x106526198>``

    .. seealso::

        | :py:func:`~verticapy.connection.current_connection` :
            Returns the current VerticaPy connection.
        | :py:func:`~verticapy.connection.new_connection` :
            Creates a new VerticaPy connection.
        | :py:func:`~verticapy.connection.set_connection` :
            Sets the VerticaPy connection.
        | :py:func:`~verticapy.connection.vertica_connection` :
            Reads the input DSN and creates a
            Vertica Database connection.
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
